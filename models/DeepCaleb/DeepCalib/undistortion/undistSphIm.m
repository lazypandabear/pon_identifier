import cv2
import numpy as np

# Optional: Deep calibration library

try:
    from deepcalib.caliblib import CameraCalibrator
    HAS_DEEPCALIB = True
except ImportError:
    HAS_DEEPCALIB = False

# Optional: GeoCalib library
try:
    from geocalib import GeoCalib
    import torch
    HAS_GEOCALIB = True
except ImportError:
    HAS_GEOCALIB = False

 def auto_calibrate_intrinsics(image: np.ndarray) -> tuple[float,float,float,float]:
    """
    Use single-image calibration (DeepCalib) to estimate fx, fy, cx, cy.
    """
    if not HAS_DEEPCALIB:
        raise ImportError("DeepCalib not installed. `pip install deepcalib`. ")
    calibrator = CameraCalibrator()
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    K_est = calibrator.estimate_intrinsics(rgb)
    fx = K_est[0,0]; fy = K_est[1,1]; cx = K_est[0,2]; cy = K_est[1,2]
    return fx, fy, cx, cy

 def geo_calibrate_intrinsics(image_path: str) -> tuple[float,float,float,float]:
    """
    Use single-image calibration (GeoCalib) to estimate fx, fy, cx, cy.
    """
    if not HAS_GEOCALIB:
        raise ImportError("GeoCalib not installed. `pip install geocalib`. ")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GeoCalib().to(device)
    # load_image loads and preprocesses internally
    tensor = model.load_image(image_path).to(device)
    result = model.calibrate(tensor)
    cam = result['camera']  # tensor [fx, fy, cx, cy]
    fx, fy, cx, cy = cam[0].item(), cam[1].item(), cam[2].item(), cam[3].item()
    return fx, fy, cx, cy

 def reconstruct_and_draw_stack(
    image: np.ndarray,
    pts2d: np.ndarray,
    cube_dim: float,
    num_stacks: int,
    num_per_stack: int,
    inner_spacing: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray = None,
    draw_axes: bool = False
) -> np.ndarray:
    """
    Reconstructs and draws a stack of cuboids (NAP boxes) onto an image.
    """
    # Define PnP reference points: positive Y down, Z-forward
    ref_3d = np.array([
        [0, 0, 0],
        [cube_dim, 0, 0],
        [cube_dim, cube_dim, 0],
        [0, 0, cube_dim]
    ], dtype=np.float64)
    success, rvec, tvec = cv2.solvePnP(ref_3d, pts2d, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_EPNP)
    if not success:
        raise RuntimeError("PnP failed.")
    out = image.copy()
    if draw_axes:
        axes = np.float32([[0,0,0],[cube_dim,0,0],[0,cube_dim,0],[0,0,cube_dim]])
        proj, _ = cv2.projectPoints(axes, rvec, tvec, camera_matrix, dist_coeffs)
        p = proj.reshape(-1,2).astype(int)
        o,x,y,z = map(tuple,p)
        cv2.line(out,o,x,(0,0,255),2); cv2.line(out,o,y,(0,255,0),2); cv2.line(out,o,z,(255,0,0),2)
    spacing = cube_dim + inner_spacing
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    for s in range(num_stacks):
        for lvl in range(num_per_stack):
            dx, dy = s*spacing, lvl*spacing
            corners = np.array([
                [dx,dy,0],[dx+cube_dim,dy,0],[dx+cube_dim,dy+cube_dim,0],[dx,dy+cube_dim,0],
                [dx,dy,cube_dim],[dx+cube_dim,dy,cube_dim],[dx+cube_dim,dy+cube_dim,cube_dim],[dx,dy+cube_dim,cube_dim]
            ], dtype=np.float64)
            proj, _ = cv2.projectPoints(corners, rvec, tvec, camera_matrix, dist_coeffs)
            pts = proj.reshape(-1,2).astype(int)
            for i0,i1 in edges:
                cv2.line(out,tuple(pts[i0]),tuple(pts[i1]),(0,255,0),1)
    return out

 def pick_coordinates(image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load {image_path}")
    def on_mouse(e,x,y,_,__):
        if e==cv2.EVENT_LBUTTONDOWN: print(f"Clicked at: ({x},{y})")
    cv2.namedWindow('Pick'); cv2.setMouseCallback('Pick',on_mouse)
    while True:
        cv2.imshow('Pick',img)
        if cv2.waitKey(1)&0xFF in (27,ord('q')): break
    cv2.destroyAllWindows()

 if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path')
    parser.add_argument('--pts', nargs=8, type=float, help='A,B,C,E coords')
    parser.add_argument('--cube_dim', type=float, default=50.0)
    parser.add_argument('--num_stacks', type=int, default=2)
    parser.add_argument('--num_per_stack', type=int, default=12)
    parser.add_argument('--inner_spacing', type=float, default=5.0)
    parser.add_argument('--fx', type=float)
    parser.add_argument('--fy', type=float)
    parser.add_argument('--cx', type=float)
    parser.add_argument('--cy', type=float)
    parser.add_argument('--draw_axes', action='store_true')
    parser.add_argument('--pick', action='store_true')
    parser.add_argument('--auto_calib', action='store_true')
    parser.add_argument('--geo_calib', action='store_true', help='Use GeoCalib to estimate intrinsics')
    args = parser.parse_args()
    if args.pick:
        pick_coordinates(args.image_path); exit(0)
    img = cv2.imread(args.image_path)
    h,w = img.shape[:2]
    if args.geo_calib:
        fx,fy,cx,cy = geo_calibrate_intrinsics(args.image_path)
        print(f"GeoCalib -> fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
    elif args.auto_calib:
        fx,fy,cx,cy = auto_calibrate_intrinsics(img)
        print(f"DeepCalib -> fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
    else:
        cx = args.cx if args.cx is not None else w/2
        cy = args.cy if args.cy is not None else h/2
        if args.fx and args.fy:
            fx,fy = args.fx,args.fy
        else:
            fx = fy = (w/2)/np.tan(np.deg2rad(60/2))
    if not args.pts:
        raise ValueError('Supply --pts or use --pick!')
    pts2d = np.array(args.pts,dtype=np.float64).reshape(4,2)
    K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])
    out_img = reconstruct_and_draw_stack(
        img,pts2d,args.cube_dim,args.num_stacks,args.num_per_stack,args.inner_spacing,K,None,args.draw_axes
    )
    cv2.imshow('Stack Reconstruction', out_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
