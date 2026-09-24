"""
P / space: pause  |  R: reset to initial scene  |  1 / 2 / 3: camera

"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from functions import MODEL_PATH,SETTINGS

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--headless',action='store_true')
    parser.add_argument('--record',type=Path)
    parser.add_argument('--seconds',type=float,default=100)
    parser.add_argument('--backend',choices=['auto','osqp','scipy'],default='auto')
    parser.add_argument('--log',type=Path,default=Path(__file__).parent/'results'/'latest_run.json')
    parser.add_argument('--width',type=int,default=1280);parser.add_argument('--height',type=int,default=800)
    args=parser.parse_args()
    try:
        import mujoco
        from functions import RobotModel
        from controller import WholeBodyController
        from controller import Sequencer,SequenceFault
        from functions import QPFailure
    except ImportError as ex:
        raise SystemExit(f'Missing dependency: {ex}. Run: python -m pip install numpy scipy osqp mujoco imageio imageio-ffmpeg pillow')
    robot=RobotModel(MODEL_PATH);controller=WholeBodyController(robot,args.backend);seq=Sequencer(robot)
    flags={'paused':False,'reset':False,'camera':0,'applied_camera':-1};frames=0;rows=[];fault=None;viewer=None;renderer=None;writer=None
    def key(code):
        if code in (32,80):flags['paused']=not flags['paused']
        elif code==82:flags['reset']=True
        elif code in (49,50,51):flags['camera']=code-49
    if not args.headless:
        import mujoco.viewer
        viewer=mujoco.viewer.launch_passive(robot.m,robot.d,key_callback=key,show_left_ui=False,show_right_ui=False)
    if args.record:
        try:
            import imageio.v2 as imageio
            renderer=mujoco.Renderer(robot.m,args.height,args.width)
            args.record.parent.mkdir(parents=True,exist_ok=True)
            writer=imageio.get_writer(str(args.record),fps=30,codec='libx264',quality=8)
        except Exception as ex:
            if viewer:viewer.close()
            raise SystemExit(f'Recording initialization failed: {ex}. For a no-display test use --headless without --record.')
    next_control=0.;next_frame=0.;next_sync=0.;last_phase=None;last_info={}
    start_wall=time.perf_counter();completed_at=None;next_diagnostic=0.
    try:
        while robot.d.time<args.seconds and (viewer is None or viewer.is_running()):
            wall=time.perf_counter()
            if flags['reset']:
                robot.reset();controller=WholeBodyController(robot,args.backend);seq=Sequencer(robot)
                next_control=next_frame=next_sync=0.;fault=None;completed_at=None;last_info={};last_phase=None
                flags['reset']=False;flags['paused']=False;rows=[];next_diagnostic=0.
            if not flags['paused'] and fault is None:
                if robot.d.time+1e-9>=next_control:
                    try:
                        last_info=controller.cycle(seq.reference(),seq.base_goal,seq.grip)
                        last_info.update(time=float(robot.d.time),phase=seq.phase)
                        seq.observe(last_info,SETTINGS.control_dt)
                        rows.append(last_info)
                        if seq.phase=='LIFT' and robot.d.time>=next_diagnostic:
                            blocked=[k for k,v in seq.guards.items() if not v]
                            print(f'LIFT blocked: {blocked} | {seq.diagnostics}',flush=True)
                            next_diagnostic=float(robot.d.time)+1.
                        if seq.phase!=last_phase:
                            print(f'{robot.d.time:6.2f}s  {seq.phase:16s}  pos={last_info["position_error"]*1000:6.1f} mm  rot={np.rad2deg(last_info["orientation_error"]):5.1f} deg',flush=True)
                            last_phase=seq.phase
                        if seq.done and completed_at is None:completed_at=float(robot.d.time)
                    except (QPFailure,SequenceFault,ValueError,RuntimeError,np.linalg.LinAlgError) as ex:
                        rows.append(dict(last_info, exception=str(ex)))
                        fault=f'{type(ex).__name__}: {ex}'
                        print('FAULT - simulation frozen for inspection:',fault,flush=True)
                        
                        flags['paused']=True
                    next_control+=SETTINGS.control_dt
                if fault is None:robot.step()
            if viewer is not None and (wall>=next_sync or flags['paused']):
                with viewer.lock():
                    if flags['applied_camera']!=flags['camera']:
                        viewer.cam.type=mujoco.mjtCamera.mjCAMERA_FIXED
                        viewer.cam.fixedcamid=flags['camera']
                        flags['applied_camera']=flags['camera']
                    viewer.opt.geomgroup[3]=1
                    
                    viewer.user_scn.ngeom=1;g=viewer.user_scn.geoms[0]
                    mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_SPHERE,np.array([.001,.001,.001]),
                                       robot.d.xpos[robot.base]+np.array([0,0,1.4]),np.eye(3).ravel(),np.array([0.,0.,0.,0.]))
                    g.label=('FAULT: press R to reset' if fault else ('PAUSED | ' if flags['paused'] else '')+seq.phase)
                viewer.sync();next_sync=wall+1/60
            if writer is not None and robot.d.time>=next_frame and fault is None:
                renderer.update_scene(robot.d,camera='overview')
                frame=renderer.render()
                from PIL import Image,ImageDraw
                im=Image.fromarray(frame);draw=ImageDraw.Draw(im)
                draw.rectangle((18,18,410,88),fill=(20,31,39))
                draw.text((32,29),'WBC / 06  |  PHYSICS SIMULATION',fill=(235,243,243))
                draw.text((32,51),f'{seq.phase}   |   t = {robot.d.time:.2f} s',fill=(72,199,214))
                writer.append_data(np.asarray(im));frames+=1;next_frame+=1/30
            if args.headless and fault:break
            if completed_at is not None and robot.d.time-completed_at>1.0:break
            if viewer is not None:
                elapsed=time.perf_counter()-wall
                time.sleep(max(.001 if flags['paused'] else 0.,robot.m.opt.timestep-elapsed))
    finally:
        if writer:writer.close()
        if renderer:renderer.close()
        if viewer:viewer.close()
        success=seq.success() and fault is None
        result={'success':success,'fault':fault,'phase':seq.phase,'sim_seconds':float(robot.d.time),
                'wall_seconds':time.perf_counter()-start_wall,'mujoco_version':mujoco.__version__,
                'phase_guards':seq.guards,'phase_diagnostics':seq.diagnostics,
                'qp_backend':controller.qp.backend,'max_object_height':seq.max_object_z,
                'final_object_position':robot.state()['object'].tolist(),
                'grasp_confirmed':seq.grasp_confirmed,'events':seq.events,'samples':rows}
        args.log.parent.mkdir(parents=True,exist_ok=True);args.log.write_text(json.dumps(result,indent=2))
        print(f'Run report: {args.log}\nSuccess: {success}; phase: {seq.phase}; fault: {fault}',flush=True)
    return 0 if success else 2

if __name__=='__main__':raise SystemExit(main())
