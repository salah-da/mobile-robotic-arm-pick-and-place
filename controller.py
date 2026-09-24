
import numpy as np
from functions import SETTINGS, HOME
from functions import (task_acceleration,dynamic_allocation,make_bounds,
                          HierarchicalQP,QPFailure,wrap)

class WholeBodyController:
    def __init__(self,robot,backend='auto',cfg=SETTINGS):
        self.robot=robot;self.cfg=cfg;self.qp=HierarchicalQP(backend)
    def cycle(self,ref,base_goal,grip):
        cfg=self.cfg;r=self.robot;s=r.state()
        if not np.all(np.isfinite(r.d.qpos)) or not np.all(np.isfinite(r.d.qvel)):
            raise QPFailure('Invalid sensor state')
        if np.any(s['q']<r.qlo) or np.any(s['q']>r.qhi):raise QPFailure('Measured arm joint limit exceeded')
        if np.max(np.abs(r.d.qvel[r.wheel_v]))>7.2:raise QPFailure('Measured wheel speed exceeded guard')
        if s['tilt']>cfg.max_tilt:raise QPFailure('Base tilt exceeded planar-model threshold')
        if abs(s['lateral_speed'])>cfg.max_lateral_speed:raise QPFailure('Lateral slip exceeded model threshold')
        M,b,B,J,jdot,S=r.dynamics(s)
        task,ep,er=task_acceleration(s['p'],s['R'],J@s['eta'],ref,cfg)
        bt=task-jdot
        a_nom,sigma,damping=dynamic_allocation(M,J,bt,s['eta'],s['q'],r.qlo,r.qhi,cfg)
        if sigma<cfg.min_normalized_sigma:raise QPFailure('Task rank/conditioning threshold exceeded')
        bounds,G,c=make_bounds(M,b,B,s['eta'],s['q'],r.qlo,r.qhi,r.umin,r.umax,cfg)
       
        P=np.eye(8)[[4]]
        posture_target=np.array([5*(.85-s['q'][2])-4*s['eta'][4]])
        dx,dy=np.array(base_goal[:2])-s['base'][:2]
      
        distance=float(np.hypot(dx,dy))
        bearing=float(np.arctan2(dy,dx))
        alpha=wrap(bearing-s['yaw'])
        direction=1.
        if abs(alpha)>np.pi/2:
            direction=-1.;bearing=wrap(bearing+np.pi)
            alpha=wrap(bearing-s['yaw'])
        beta=wrap(base_goal[2]-bearing)
        v_goal=direction*min(.20,.8*distance)*max(0.,np.cos(alpha))
        w_goal=np.clip(2.*alpha-.6*beta,-.5,.5)
        if distance<.025:
            v_goal=0.;w_goal=np.clip(2.*wrap(base_goal[2]-s['yaw']),-.5,.5)
        base_acc=np.array([3.*(v_goal-s['eta'][0]),3.*(w_goal-s['eta'][1])])
        # Base positioning uses both directions left by the six-dimensional tool task.
        Cbase=np.eye(8)[:2]/np.array([.6,1.3])[:,None]
        a,info=self.qp.solve(J,bt,bounds,a_nom,(P,posture_target),
                             (Cbase,base_acc/np.array([.6,1.3])),G,c,r.umax,cfg.task_length)
        u=G@a+c
        inverse_residual=float(np.linalg.norm(M@a+b-B@u,np.inf))
        if inverse_residual>1e-6:raise QPFailure('Inverse dynamics consistency failed')
        if info['solve_seconds']>cfg.max_solver_walltime:raise QPFailure('Controller deadline guard exceeded')
        r.command(u,grip)
        info.update(base_position=s['base'][:2].tolist(),base_goal=list(base_goal),
                    base_position_error=float(np.hypot(dx,dy)),
                    base_linear_speed=float(s['eta'][0]),base_yaw_speed=float(s['eta'][1]),
                    position_error=ep,orientation_error=er,sigma_min=sigma,
                    damping=damping,inverse_dynamics_residual=inverse_residual,
                    nominal_allocation_difference=float(np.linalg.norm(a-a_nom)),
                    torque=u.tolist(),phase='',a=a.tolist(),lateral_speed=s['lateral_speed'],tilt=s['tilt'])
        return info



import numpy as np
from functions import SETTINGS,PICK,PLACE,R_DOWN,TABLE_TOP,OBJECT_HALF
from functions import PoseSegment

class SequenceFault(RuntimeError):pass

class Sequencer:
    def __init__(self,robot,cfg=SETTINGS):
        self.robot=robot;self.cfg=cfg;self.reset()
    def reset(self):
        s=self.robot.state();self.index=-1;self.events=[];self.dwell=0.;self.contact_since=None
        self.grasp_confirmed=False;self.done=False;self.grip=self.cfg.grip_open
        self.max_object_z=s['object'][2];self.released=False
        self.guards={};self.diagnostics={};self.grasp_height=None
        # Preplanned obstacle-clearance waypoints, not a general collision planner.
        raised=s['p'].copy();raised[2]=.78
        self.steps=[
          ('READY',s['p'].copy(),s['R'].copy(),1.,[0,0,0]),
          ('RAISE',raised,R_DOWN,3.,[0,0,0]),
          ('APPROACH',PICK+np.array([0,0,.115]),R_DOWN,6.,[.52,.38,0]),
          ('ALIGN',PICK,R_DOWN,3.,[.52,.38,0]),
          ('CLOSE',PICK,R_DOWN,1.2,[.52,.38,0]),
          ('LIFT',PICK+np.array([0,0,.155]),R_DOWN,3.5,[.54,.38,0]),
          ('CLEAR_STATION',np.array([1.06,-.12,.86]),R_DOWN,5.,[.56,-.18,-.35]),
          ('TRANSPORT',np.array([1.46,-.48,.86]),R_DOWN,5.,[.94,-.48,0]),
          ('PRE_PLACE',PLACE+np.array([0,0,.115]),R_DOWN,5.,[1.32,-.48,0]),
          ('PLACE',PLACE,R_DOWN,3.,[1.32,-.48,0]),
          ('RELEASE',PLACE,R_DOWN,1.5,[1.32,-.48,0]),
          ('RETREAT',PLACE+np.array([0,0,.15]),R_DOWN,3.5,[1.34,-.48,0]),
        ]
        self._advance()
    def _advance(self):
        self.index+=1;self.dwell=0.;self.contact_since=None
        if self.index>=len(self.steps):
            self.done=True;self.phase='COMPLETE';return
        self.phase,goal,R,T,self.base_goal=self.steps[self.index]
        s=self.robot.state();self.start=float(self.robot.d.time)
        if self.phase=='PLACE':
            # Nest top is TABLE_TOP + .003; target object support height,
            # accounting for where the fingers actually gripped the object.
            self.place_tool_goal=np.array([PLACE[0],PLACE[1],TABLE_TOP+.003+OBJECT_HALF])+s['p']-s['object']
            goal=self.place_tool_goal.copy()
        elif self.phase=='RELEASE':
            goal=self.place_tool_goal.copy()
        self.segment=PoseSegment(s['p'],s['R'],goal,R,T)
        self.events.append({'time':self.start,'phase':self.phase})
        if self.phase=='CLOSE':self.grip=self.cfg.grip_closed
        if self.phase=='RELEASE':
            self.grip=self.cfg.grip_open;self.robot.detach_payload_model();self.released=True
    def reference(self):return self.segment.sample(self.robot.d.time-self.start)
    def observe(self,info,dt):
        if self.done:return
        r=self.robot;s=r.state();normal,support,collision=r.contacts()
        t=float(r.d.time);elapsed=t-self.start
        self.max_object_z=max(self.max_object_z,s['object'][2])
        if collision:raise SequenceFault('Unexpected robot-table contact')
        if s['object'][2]<.56:raise SequenceFault('Object dropped below station height')
        if self.grasp_confirmed and self.phase not in ('PLACE','RELEASE','RETREAT','COMPLETE'):
            if np.linalg.norm(s['object']-s['p'])>.075:
                raise SequenceFault('Object lost from gripper')
        # Carrying phases allow modest orientation error.
        carry_phases = {'LIFT', 'CLEAR_STATION', 'TRANSPORT', 'PRE_PLACE'}
        rotation_tolerance = np.deg2rad(
            6.0 if self.phase in carry_phases else 3.72
        )

        pose_ok = (
            info['position_error'] < .012
            and info['orientation_error'] < rotation_tolerance
        )
        if self.phase in ('ALIGN','PLACE'):pose_ok=info['position_error']<.005 and info['orientation_error']<.035
        speed_ok=np.linalg.norm(s['eta'][2:])<.20 and abs(s['eta'][0])<.045 and abs(s['eta'][1])<.10
        ready=elapsed>=self.segment.duration and pose_ok and speed_ok
        if self.phase=='CLOSE':
            two_sided=bool(np.all(normal>1.0))
            ready=ready and two_sided
            if elapsed>self.cfg.close_timeout and not two_sided:raise SequenceFault('Two-sided grasp was not confirmed')
      
        if self.phase=='LIFT':
            tcp_v,tcp_w,obj_v=r.measured_motion()
            baseline=PICK[2] if self.grasp_height is None else self.grasp_height
            height_gain=float(s['object'][2]-baseline)
            goal_position=self.segment.p0+self.segment.dp
            endpoint_error=float(np.linalg.norm(goal_position-s['p']))
            self.guards={
                'trajectory_finished':bool(elapsed>=self.segment.duration),
                'endpoint_position':bool(endpoint_error<.012),
                'endpoint_orientation':bool(info['orientation_error']<rotation_tolerance),
                'tool_linear_settled':bool(np.linalg.norm(tcp_v)<.035),
                'tool_angular_settled':bool(np.linalg.norm(tcp_w)<.10),
                'object_settled':bool(np.linalg.norm(obj_v)<.04),
                'base_linear_settled':bool(abs(s['eta'][0])<.045),
                'base_yaw_settled':bool(abs(s['eta'][1])<.10),
                'lift_height':bool(height_gain>.10),
                'clear_of_table':bool(not support),
                'object_near_tool':bool(np.linalg.norm(s['object']-s['p'])<.075),
            }
            self.diagnostics={
                'height_gain_m':height_gain,'endpoint_error_m':endpoint_error,
                'orientation_error_rad':float(info['orientation_error']),
                'tcp_speed_m_s':float(np.linalg.norm(tcp_v)),
                'tcp_angular_speed_rad_s':float(np.linalg.norm(tcp_w)),
                'object_speed_m_s':float(np.linalg.norm(obj_v)),
                'arm_joint_speed_norm':float(np.linalg.norm(s['eta'][2:])),
                'pad_normal_force_N':normal.tolist(),
            }
            info['lift_guards']=self.guards.copy()
            info['lift_diagnostics']=self.diagnostics.copy()
            ready=all(self.guards.values())
        if self.phase!='LIFT':
            self.guards={
                'trajectory_finished':bool(elapsed>=self.segment.duration),
                'tool_pose':bool(pose_ok),
                'arm_settled':bool(np.linalg.norm(s['eta'][2:])<.20),
                'base_linear_settled':bool(abs(s['eta'][0])<.045),
                'base_yaw_settled':bool(abs(s['eta'][1])<.10),
            }
            if self.phase=='CLOSE':self.guards['two_sided_grasp']=two_sided
            if self.phase in ('PLACE','RELEASE'):self.guards['table_support']=bool(support)
            if self.phase=='RELEASE':
                self.guards['object_at_place']=bool(np.linalg.norm(s['object'][:2]-PLACE[:2])<.03)
                self.guards['fingers_open']=bool(np.all(r.d.qpos[r.finger_q]>.029))
            self.diagnostics={
                'phase':self.phase,
                'endpoint_error_m':float(np.linalg.norm(self.segment.p0+self.segment.dp-s['p'])),
                'orientation_error_rad':float(info['orientation_error']),
                'arm_joint_speed_norm':float(np.linalg.norm(s['eta'][2:])),
                'base_position_m':s['base'][:2].tolist(),
                'base_goal_m':list(self.base_goal[:2]),
                'base_position_error_m':float(np.linalg.norm(np.array(self.base_goal[:2])-s['base'][:2])),
                'base_linear_speed_m_s':float(s['eta'][0]),
                'base_yaw_speed_rad_s':float(s['eta'][1]),
            }
        info['phase_guards']=self.guards.copy()
        info['phase_diagnostics']=self.diagnostics.copy()
        if self.phase=='PLACE':ready=ready and support
        if self.phase=='RELEASE':
            ready=ready and support and np.linalg.norm(s['object'][:2]-PLACE[:2])<.03
            ready=ready and np.all(r.d.qpos[r.finger_q]>.029)
        self.dwell=self.dwell+dt if ready else 0.
        if self.dwell>.25:
            if self.phase=='CLOSE':
                self.grasp_confirmed=True;self.grasp_height=float(s['object'][2]);r.attach_payload_model()
            self._advance()
        elif elapsed>self.segment.duration+self.cfg.max_phase_extra:
            blocked=[name for name,ok in self.guards.items() if not ok]
            raise SequenceFault(f'{self.phase}: timeout; blocked={blocked}; measurements={self.diagnostics}')
    def success(self):
        s=self.robot.state();normal,support,_=self.robot.contacts()
        return bool(self.done and self.grasp_confirmed and self.released and support and
                    self.max_object_z>PICK[2]+.10 and np.linalg.norm(s['object'][:2]-PLACE[:2])<.03)
