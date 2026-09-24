
from dataclasses import dataclass
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / 'workcell.xml'
ARM_JOINTS = ('q1_shoulder_pan','q2_shoulder_lift','q3_elbow',
              'q4_wrist1','q5_wrist2','q6_wrist3')
ARM_ACTUATORS = tuple(f'tau_{i}' for i in range(1,7))
WHEELS = ('wheel_left_j','wheel_right_j')
FINGERS = ('finger_left_j','finger_right_j')
HOME = np.array([0., -1.20, 1.60, 0., 1.17, 0.])
R_DOWN = np.array([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]])
PICK = np.array([1.10,.38,.686])
PLACE = np.array([1.90,-.48,.686])
TABLE_TOP = .650
OBJECT_HALF = .035

@dataclass
class Settings:
    dt: float = .002
    control_dt: float = .010
    wheel_radius: float = .10
    half_track: float = .25
    task_length: float = .30
    coordinate_length: float = .30
    reference_mass: float = 1.
    kp_position: float = 45.
    kd_position: float = 16.
    kp_rotation: float = 24.
    kd_rotation: float = 10.
    cbf_k1: float = 3.
    cbf_k2: float = 5.
    velocity_barrier_gain: float = 5.
    joint_margin: float = .045
    min_normalized_sigma: float = .001
    max_tilt: float = .12
    max_lateral_speed: float = .06
    max_solver_walltime: float = .25  # fault guard, NOT a real-time guarantee
    nominal_rank_damping: float = .015
    max_phase_extra: float = 12.
    grip_open: float = .035
    grip_closed: float = 0.
    close_timeout: float = 4.

SETTINGS = Settings()

# ====================================================================
# control_math.py

from dataclasses import dataclass
import time
import numpy as np
from scipy.linalg import qr, null_space
from scipy.optimize import minimize, LinearConstraint
from scipy import sparse

def skew(v):
    x,y,z=v
    return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])

def vee(M): return np.array([M[2,1],M[0,2],M[1,0]])
def wrap(a): return (a+np.pi)%(2*np.pi)-np.pi

def task_acceleration(p,R,V,ref,cfg):
    
    ep=ref.p-p
    Q=ref.R.T@R
    eR=.5*vee(Q-Q.T)                         
    ev=ref.v-V[:3]
    transported=R.T@ref.R@ref.omega
    eomega=V[3:]-transported
    ap=ref.a+cfg.kd_position*ev+cfg.kp_position*ep
    ar=R.T@ref.R@ref.alpha-skew(V[3:])@transported-cfg.kd_rotation*eomega-cfg.kp_rotation*eR
    angle=np.arccos(np.clip((np.trace(Q)-1)/2,-1,1))
    return np.r_[ap,ar],float(np.linalg.norm(ep)),float(angle)

def dynamic_allocation(M,J,b_task,eta,q,qmin,qmax,cfg):
   
    TL=np.diag([1,1,1,cfg.task_length,cfg.task_length,cfg.task_length])
    JL=TL@J
    K=JL@np.linalg.solve(M,JL.T)
    eig=np.linalg.eigvalsh(cfg.reference_mass*K)
    sigma=float(np.sqrt(max(eig[0],0.)))
    # This is the equivalent normalized inverse from Section 8, eq 25.
    damping=cfg.nominal_rank_damping*max(0.,1-sigma/.025)
    inverse=np.linalg.solve(M,JL.T)@np.linalg.solve(K+damping*damping/cfg.reference_mass*np.eye(6),TL)
    N=np.eye(8)-inverse@J
    z=-2*eta
    # Bounded torque preference converted to acceleration (Section 17).
    margin=.20; gain=5.
    force=np.zeros(8)
    force[2:]=gain*(np.clip((qmin+margin-q)/margin,0,1)-np.clip((q-(qmax-margin))/margin,0,1))
    z+=np.linalg.solve(M,force)
    return inverse@b_task+N@z, sigma, damping

@dataclass
class LinearBounds:
    A: np.ndarray
    lo: np.ndarray
    hi: np.ndarray

    def residual(self,x):
        y=self.A@x
        return float(max(0.,np.max(self.lo-y),np.max(y-self.hi)))

def make_bounds(M,b,B,eta,q,qlo,qhi,umin,umax,cfg):
   
    G=np.linalg.solve(B,M);c=np.linalg.solve(B,b)
    A=[G];lower=[umin-c];upper=[umax-c]
    # Conservative joint margins are explicit design parameters.
    lo=qlo+cfg.joint_margin;hi=qhi-cfg.joint_margin
    k1,k2=cfg.cbf_k1,cfg.cbf_k2
    P=np.eye(8)[2:]
    amin=-(k1+k2)*eta[2:]-k1*k2*(q-lo)
    amax=-(k1+k2)*eta[2:]+k1*k2*(hi-q)
    A.append(P);lower.append(amin);upper.append(amax)
    # Continuous-time velocity CBFs (not a position test masquerading as one).
    vmax=np.array([.30,.65,1.,1.,1.,1.3,1.3,1.5])
    alpha=cfg.velocity_barrier_gain
    A.append(np.eye(8));lower.append(alpha*(-vmax-eta));upper.append(alpha*(vmax-eta))
    D=np.zeros((2,8));D[:,:2]=np.array([[1,-cfg.half_track],[1,cfg.half_track]])/cfg.wheel_radius
    w=D@eta;wmax=6.
    A.append(D);lower.append(alpha*(-wmax-w));upper.append(alpha*(wmax-w))
    # Explicit acceleration limits, normalized in optimizer variable coordinates.
    alim=np.array([.60,1.3,3.,3.,3.,4.,4.,5.])
    A.append(np.eye(8));lower.append(-alim);upper.append(alim)
    return LinearBounds(np.vstack(A),np.concatenate(lower),np.concatenate(upper)),G,c

class QPFailure(RuntimeError): pass

class HierarchicalQP:
    
    def __init__(self,backend='auto'):
        self.last=np.zeros(8)
        try:
            if backend=='scipy': raise ImportError
            import osqp
            self.osqp=osqp
        except ImportError:
            if backend=='osqp': raise
            self.osqp=None
        self.backend='osqp' if self.osqp else 'scipy'

    def _solve(self,C,target,bounds,E,e,x0):
      
        scales=np.array([.60,1.3,3.,3.,3.,4.,4.,5.])
        D=np.diag(scales)
        Cn=C@D;An=bounds.A@D;En=E@D
       
        if len(En):
            _,R,piv=qr(En.T,mode='economic',pivoting=True)
            rank=np.linalg.matrix_rank(R,tol=1e-9)
            ids=piv[:rank];En=En[ids];ee=e[ids]
        else: ee=np.empty(0)
        stacked=np.vstack([An,En]);lo=np.r_[bounds.lo,ee];hi=np.r_[bounds.hi,ee]
        norm=np.maximum(np.linalg.norm(stacked,axis=1),1e-8)
        stacked=stacked/norm[:,None];lo=lo/norm;hi=hi/norm
        H=Cn.T@Cn;g=-Cn.T@target
        if self.osqp:
            solver=self.osqp.OSQP()
            solver.setup(P=sparse.csc_matrix(np.triu(H)),q=g,A=sparse.csc_matrix(stacked),l=lo,u=hi,
                         verbose=False,eps_abs=1e-6,eps_rel=1e-6,max_iter=10000,polishing=True)
            solver.warm_start(x=x0/scales)
            result=solver.solve()
            if result.x is None or result.info.status_val not in (1,2):
                raise QPFailure('OSQP: '+result.info.status)
            answer=scales*result.x
        else:
           
            eq=np.isfinite(lo)&np.isfinite(hi)&(np.abs(hi-lo)<1e-12)
            cons=[]
            if np.any(~eq):cons.append(LinearConstraint(stacked[~eq],lo[~eq],hi[~eq]))
            if np.any(eq):cons.append(LinearConstraint(stacked[eq],lo[eq],hi[eq]))
            result=minimize(lambda x:.5*np.dot(Cn@x-target,Cn@x-target),x0/scales,
                jac=lambda x:H@x+g,method='SLSQP',constraints=cons,
                options={'ftol':1e-11,'maxiter':160})
            if not result.success:raise QPFailure('SLSQP: '+result.message)
            answer=scales*result.x
        if not np.all(np.isfinite(answer)) or bounds.residual(answer)>2e-4:
            raise QPFailure('Hard-constraint residual exceeds acceptance tolerance')
        if len(E) and np.max(np.abs(E@answer-e))>2e-4:
            raise QPFailure('Higher-priority output changed')
        return answer

    def solve(self,J,bt,bounds,a_nom,posture,base,G,c,umax,task_length):
        start=time.perf_counter();x=a_nom.copy() if bounds.residual(a_nom)<1e-6 else self.last.copy();E=np.empty((0,8));e=np.empty(0)
        TL=np.diag([1,1,1,task_length,task_length,task_length])
        levels=[(TL@J,TL@bt),base,posture,
                (G/umax[:,None],-c/umax)]
        achieved=[]
        for C,t in levels:
            if C.size==0:continue
            
            if np.linalg.matrix_rank(E,tol=1e-8)==8:break
            
            try:
                candidate = self._solve(C, t, bounds, E, e, x)
            except QPFailure as exc:
                level = len(achieved) + 1
                if not achieved:
                    raise QPFailure(f"HQP level {level}: {exc}") from exc
                break
            else:
                x= candidate
            achieved.append(float(np.linalg.norm(C@x-t)))
            E=np.vstack([E,C]);e=np.r_[e,C@x]
        self.last=x
        return x,{'solve_seconds':time.perf_counter()-start,'level_residuals':achieved,
                  'constraint_residual':bounds.residual(x),'task_slack':(J@x-bt).tolist()}

# ====================================================================
# trajectory.py

from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation

@dataclass
class Reference:
    p: np.ndarray
    R: np.ndarray
    v: np.ndarray
    omega: np.ndarray
    a: np.ndarray
    alpha: np.ndarray

def blend(t,T):
    if t<=0:return 0.,0.,0.
    if t>=T:return 1.,0.,0.
    s=t/T
    return 10*s**3-15*s**4+6*s**5,(30*s**2-60*s**3+30*s**4)/T,(60*s-180*s**2+120*s**3)/T**2

class PoseSegment:
    def __init__(self,p0,R0,p1,R1,duration):
        self.p0=np.array(p0);self.R0=np.array(R0);self.dp=np.array(p1)-p0
        self.rotvec=Rotation.from_matrix(self.R0.T@R1).as_rotvec()
        self.duration=float(duration)
    def sample(self,t):
        s,sd,sdd=blend(t,self.duration)
        return Reference(self.p0+s*self.dp,self.R0@Rotation.from_rotvec(s*self.rotvec).as_matrix(),
                         sd*self.dp,sd*self.rotvec,sdd*self.dp,sdd*self.rotvec)

# ====================================================================
# model.py

import numpy as np

class RobotModel:
    def __init__(self,path,cfg=SETTINGS):
        global mujoco
        import mujoco
        self.cfg=cfg
        self.m=mujoco.MjModel.from_xml_path(str(path));self.d=mujoco.MjData(self.m)
        self.nom=mujoco.MjData(self.m);self.plus=mujoco.MjData(self.m);self.minus=mujoco.MjData(self.m)
        self.base_joint=self.m.joint('base_free').id
        self.base_q=int(self.m.jnt_qposadr[self.base_joint]);self.base_v=int(self.m.jnt_dofadr[self.base_joint])
        self.arm_q=np.array([self.m.jnt_qposadr[self.m.joint(n).id] for n in ARM_JOINTS])
        self.arm_v=np.array([self.m.jnt_dofadr[self.m.joint(n).id] for n in ARM_JOINTS])
        self.wheel_v=np.array([self.m.jnt_dofadr[self.m.joint(n).id] for n in WHEELS])
        self.finger_q=np.array([self.m.jnt_qposadr[self.m.joint(n).id] for n in FINGERS])
        self.finger_v=np.array([self.m.jnt_dofadr[self.m.joint(n).id] for n in FINGERS])
        self.robot_act=np.array([self.m.actuator(n).id for n in ('tau_L','tau_R')+ARM_ACTUATORS])
        self.grip_act=np.array([self.m.actuator('gripper_'+s).id for s in ['left','right']])
        self.site=self.m.site('ee_site').id;self.base=self.m.body('base_link').id
        self.obj=self.m.body('pick_object').id;self.objgeom=self.m.geom('part_collision').id
        self.pads=[self.m.geom('pad_'+s).id for s in ['left','right']]
        self.table_geoms={self.m.geom(n).id for n in ['pick_top','pick_nest','place_top','place_nest']}
        self.qlo=np.array([self.m.joint(n).range[0] for n in ARM_JOINTS])
        self.qhi=np.array([self.m.joint(n).range[1] for n in ARM_JOINTS])
        self.umin=self.m.actuator_ctrlrange[self.robot_act,0].copy()
        self.umax=self.m.actuator_ctrlrange[self.robot_act,1].copy()
        self.fullM=np.zeros((self.m.nv,self.m.nv))
        self.Bfull=np.zeros((self.m.nv,8))
        for col,name in enumerate(WHEELS+ARM_JOINTS):
            aid=self.robot_act[col];jid=self.m.joint(name).id
            if int(self.m.actuator_trnid[aid,0])!=jid:raise ValueError('Actuator/joint ordering mismatch')
            self.Bfull[self.m.jnt_dofadr[jid],col]=self.m.actuator_gear[aid,0]
        self.payload=None
        self.reset()
       
        try:
            mujoco.mj_fullM(self.m,self.d,self.fullM)
            self.mass_matrix=lambda data:mujoco.mj_fullM(self.m,data,self.fullM)
        except TypeError:
            self.mass_matrix=lambda data:mujoco.mj_fullM(self.m,self.fullM,data.qM)
            self.mass_matrix(self.d)

    def reset(self):
        mujoco.mj_resetData(self.m,self.d)
        self.d.qpos[self.arm_q]=HOME;self.d.qpos[self.finger_q]=self.cfg.grip_open
        self.d.ctrl[self.grip_act]=self.cfg.grip_open
        self.payload=None;mujoco.mj_forward(self.m,self.d)

    def rotation(self,data,body=None):return data.xmat[self.base if body is None else body].reshape(3,3).copy()
    def yaw(self,data):
        R=self.rotation(data);return float(np.arctan2(R[1,0],R[0,0]))
    def basis(self,data):
        th=self.yaw(data);R=self.rotation(data);S=np.zeros((self.m.nv,8))
        S[self.base_v:self.base_v+3,0]=[np.cos(th),np.sin(th),0]
        # MuJoCo freejoint angular rates are body-frame components.
        S[self.base_v+3:self.base_v+6,1]=R.T@np.array([0.,0.,1.])
        S[self.wheel_v,:2]=np.array([[1,-self.cfg.half_track],[1,self.cfg.half_track]])/self.cfg.wheel_radius
        S[self.arm_v,2:]=np.eye(6)
        return S
    def state(self):
        mujoco.mj_forward(self.m,self.d)
        d=self.d;th=self.yaw(d);R=self.rotation(d)
        vb=d.qvel[self.base_v:self.base_v+3]
        omega_world=R@d.qvel[self.base_v+3:self.base_v+6]
        eta=np.r_[vb[0]*np.cos(th)+vb[1]*np.sin(th),omega_world[2],d.qvel[self.arm_v]]
        return dict(q=d.qpos[self.arm_q].copy(),eta=eta,
                    p=d.site_xpos[self.site].copy(),R=d.site_xmat[self.site].reshape(3,3).copy(),
                    base=d.xpos[self.base].copy(),yaw=th,
                    tilt=float(np.arccos(np.clip(R[2,2],-1,1))),
                    lateral_speed=float(-vb[0]*np.sin(th)+vb[1]*np.cos(th)),
                    object=d.xpos[self.obj].copy())
    def measured_motion(self):
        
        jp=np.zeros((3,self.m.nv));jr=np.zeros_like(jp)
        mujoco.mj_jacSite(self.m,self.d,jp,jr,self.site)
        objp=np.zeros_like(jp);objr=np.zeros_like(jp)
        mujoco.mj_jacBodyCom(self.m,self.d,objp,objr,self.obj)
        return jp@self.d.qvel, jr@self.d.qvel, objp@self.d.qvel

    def jacobian(self,data,S):
        jp=np.zeros((3,self.m.nv));jr=np.zeros_like(jp)
        mujoco.mj_jacSite(self.m,data,jp,jr,self.site)
        R=data.site_xmat[self.site].reshape(3,3)
        return np.vstack([jp,R.T@jr])@S
    def _payload_jac(self,data,J):
        mass,Ibody,r_local,Rrel=self.payload
        R=data.site_xmat[self.site].reshape(3,3);r=R@r_local
        Jp=J[:3]-skew(r)@R@J[3:]
        Jo=Rrel.T@J[3:]
        return Jp,Jo
    def attach_payload_model(self):
        
        s=self.state();Robj=self.rotation(self.d,self.obj)
        self.payload=(float(self.m.body_mass[self.obj]),np.diag(self.m.body_inertia[self.obj]),
                      s['R'].T@(s['object']-s['p']),s['R'].T@Robj)
    def detach_payload_model(self):self.payload=None
    def dynamics(self,s):
        
        cfg=self.cfg;eta=s['eta'];m=self.m
        self.nom.qpos[:]=self.d.qpos;self.nom.qvel[:]=0
        mujoco.mj_forward(m,self.nom);S=self.basis(self.nom)
        self.nom.qvel[:]=S@eta;mujoco.mj_forward(m,self.nom)
        J=self.jacobian(self.nom,S)
        eps=2e-5
        for data,sign in [(self.plus,1),(self.minus,-1)]:
            data.qpos[:]=self.nom.qpos;data.qvel[:]=S@eta
            mujoco.mj_integratePos(m,data.qpos,data.qvel,sign*eps)
            mujoco.mj_forward(m,data)
        Sp=self.basis(self.plus);Sm=self.basis(self.minus)
        Sd=(Sp-Sm)/(2*eps)
        Jp=self.jacobian(self.plus,Sp);Jm=self.jacobian(self.minus,Sm)
        Jd=(Jp-Jm)/(2*eps)
        self.mass_matrix(self.nom)
        M=S.T@self.fullM@S
        
        b=S.T@(self.nom.qfrc_bias-self.nom.qfrc_passive+self.fullM@Sd@eta)
        if self.payload is not None:
            mass,Ibody,_,_=self.payload
            Lp,Lo=self._payload_jac(self.nom,J)
            Lpp,Lop=self._payload_jac(self.plus,Jp);Lpm,Lom=self._payload_jac(self.minus,Jm)
            Lpd=(Lpp-Lpm)/(2*eps);Lod=(Lop-Lom)/(2*eps)
            Om=Lo@eta
            M+=mass*Lp.T@Lp+Lo.T@Ibody@Lo
            b+=Lp.T@(mass*(Lpd@eta-m.opt.gravity))+Lo.T@(Ibody@Lod@eta+np.cross(Om,Ibody@Om))
        M=.5*(M+M.T)
        np.linalg.cholesky(M)             # reject invalid model; no silent inertia fudge
        B=S.T@self.Bfull
        if np.linalg.matrix_rank(B)!=8:raise RuntimeError('Reduced actuator map lost rank')
        return M,b,B,J,Jd@eta,S
    def contacts(self):
        normal=np.zeros(2);support=False;unexpected=False
        for k in range(self.d.ncon):
            con=self.d.contact[k];ids={int(con.geom1),int(con.geom2)}
            if self.objgeom in ids:
                wrench=np.zeros(6);mujoco.mj_contactForce(self.m,self.d,k,wrench)
                for i,pad in enumerate(self.pads):
                    if pad in ids:normal[i]+=max(0.,wrench[0])
                if ids&self.table_geoms:support=True
            if ids&self.table_geoms:
                others=ids-self.table_geoms
                if any(self.m.geom_bodyid[g]!=self.obj and self.m.geom_bodyid[g]!=0 for g in others):unexpected=True
        return normal,support,unexpected
    def command(self,u,grip):
       
        if np.any(u<self.umin-2e-4) or np.any(u>self.umax+2e-4):raise RuntimeError('Rejected out-of-bounds torque')
        self.d.ctrl[self.robot_act]=u
        self.d.ctrl[self.grip_act]=grip
    def step(self):mujoco.mj_step(self.m,self.d)
