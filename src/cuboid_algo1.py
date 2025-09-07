import cv2
import numpy as np
import sys

# Optional GeoCalib for auto intrinsics
try:
    from geocalib import GeoCalib
    import torch
    HAS_GEOCALIB = True
except ImportError:
    HAS_GEOCALIB = False


def geo_calibrate_intrinsics(image_path: str) -> tuple[float, float, float, float]:
    """
    Estimate fx,fy,cx,cy using GeoCalib from an image.
    """
    if not HAS_GEOCALIB:
        raise ImportError(
            "GeoCalib not available. Install via `pip install -e \"git+https://github.com/cvg/GeoCalib#egg=geocalib\"`"
        )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GeoCalib().to(device)
    tensor = model.load_image(image_path).to(device)
    result = model.calibrate(tensor)
    cam = result['camera']
    K_val = cam.K.detach().cpu().numpy()[0] if hasattr(cam.K, 'detach') else np.array(cam.K)
    fx, fy = float(K_val[0,0]), float(K_val[1,1])
    cx, cy = float(K_val[0,2]), float(K_val[1,2])
    return fx, fy, cx, cy


def pick_coordinates(image_path: str):
    """
    Opens the image and prints coordinates on left-click.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load {image_path}")
    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"Clicked at: ({x}, {y})")
    cv2.namedWindow('Pick Coords')
    cv2.setMouseCallback('Pick Coords', on_mouse)
    while True:
        cv2.imshow('Pick Coords', img)
        if cv2.waitKey(1) & 0xFF in (27, ord('q')):
            break
    cv2.destroyAllWindows()


def reconstruct_and_draw_stack(
    image: np.ndarray,
    pts2d: np.ndarray,
    cube_dim: float,
    num_stacks: int,
    num_per_stack: int,
    inner_spacing: float,
    K: np.ndarray,
    dist_coeffs: np.ndarray = None,
    draw_axes: bool = False
) -> np.ndarray:
    """
    Reconstructs and draws perspective-stacked cuboids using PnP.
    """
    # 3D reference: A,B,C,E
    ref_3d = np.array([
        [0, 0, 0],
        [cube_dim, 0, 0],
        [cube_dim, cube_dim, 0],
        [0, 0, -cube_dim]
    ], dtype=np.float64)
    success, rvec, tvec = cv2.solvePnP(
        ref_3d, pts2d, K, dist_coeffs, flags=cv2.SOLVEPNP_EPNP
    )
    if not success:
        raise RuntimeError("PnP failed; check 2D points and intrinsics.")
    out = image.copy()
    # Draw reference points
    labels = ['A','B','C','E']
    for i, p in enumerate(pts2d.astype(int)):
        cv2.circle(out, tuple(p), 5, (0,0,255), -1)
        cv2.putText(out, labels[i], (p[0]+5,p[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),1)
    # Debug axes
    if draw_axes:
        axes = np.float32([[0,0,0],[cube_dim,0,0],[0,cube_dim,0],[0,0,-cube_dim]])
        proj,_ = cv2.projectPoints(axes, rvec, tvec, K, dist_coeffs)
        pts = proj.reshape(-1,2).astype(int)
        o,x_pt,y_pt,z_pt = map(tuple, pts)
        cv2.line(out,o,x_pt,(0,0,255),2)
        cv2.line(out,o,y_pt,(0,255,0),2)
        cv2.line(out,o,z_pt,(255,0,0),2)
    # Draw cubes
    spacing = cube_dim + inner_spacing
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    for s in range(num_stacks):
        for lvl in range(num_per_stack):
            dx, dy = s*spacing, lvl*spacing
            corners = np.array([
                [dx, dy, 0], [cube_dim+dx, dy, 0], [cube_dim+dx, cube_dim+dy, 0], [dx, cube_dim+dy, 0],
                [dx, dy, -cube_dim], [cube_dim+dx, dy, -cube_dim], [cube_dim+dx, cube_dim+dy, -cube_dim], [dx, cube_dim+dy, -cube_dim]
            ], dtype=np.float64)
            proj,_ = cv2.projectPoints(corners, rvec, tvec, K, dist_coeffs)
            pts = proj.reshape(-1,2).astype(int)
            for i0,i1 in edges:
                cv2.line(out, tuple(pts[i0]), tuple(pts[i1]), (0,255,0),1)
    return out

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='PnP-based cuboid stack')
    parser.add_argument('image_path', help='Image file')
    parser.add_argument('--pick', action='store_true', help='Choose 2D points')
    parser.add_argument('--pts', nargs=8, type=float, metavar=('xA','yA','xB','yB','xC','yC','xE','yE'), help='Points A,B,C,E')
    parser.add_argument('--cube_dim', type=float, default=50, help='Cube edge (mm)')
    parser.add_argument('--num_stacks', type=int, default=2, help='Number of side-by-side stacks')
    parser.add_argument('--num_per_stack', type=int, default=12, help='Cubes per stack')
    parser.add_argument('--inner_spacing', type=float, default=5, help='Gap between cubes (mm)')
    parser.add_argument('--fx', type=float, help='Focal fx (px)')
    parser.add_argument('--fy', type=float, help='Focal fy (px)')
    parser.add_argument('--cx', type=float, help='Principal cx (px)')
    parser.add_argument('--cy', type=float, help='Principal cy (px)')
    parser.add_argument('--geo_calib', action='store_true', help='Auto-estimate intrinsics with GeoCalib')
    parser.add_argument('--draw_axes', action='store_true', help='Overlay coordinate axes')
    args = parser.parse_args()

    if args.pick:
        pick_coordinates(args.image_path)
        sys.exit(0)

    # Intrinsics
    if args.geo_calib:
        fx, fy, cx, cy = geo_calibrate_intrinsics(args.image_path)
        print(f"GeoCalib -> fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
    else:
        if not all([args.fx, args.fy, args.cx, args.cy]):
            parser.error('Provide --fx/--fy/--cx/--cy or --geo_calib')
        fx, fy, cx, cy = args.fx, args.fy, args.cx, args.cy

    if not args.pts:
        parser.error('Need --pts or --pick')

    img = cv2.imread(args.image_path)
    if img is None:
        parser.error(f"Cannot load {args.image_path}")

    pts2d = np.array(args.pts, dtype=np.float64).reshape(4,2)
    K = np.array([[fx,0,cx], [0,fy,cy], [0,0,1]])

    out = reconstruct_and_draw_stack(
        img, pts2d, args.cube_dim, args.num_stacks,
        args.num_per_stack, args.inner_spacing, K, None, args.draw_axes
    )

    cv2.imshow('Stack Reconstruction', out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    if args.pick:
        pick_coordinates(args.image_path)
        sys.exit(0)
    # Intrinsics
    if args.geo_calib:
        fx, fy, cx, cy = geo_calibrate_intrinsics(args.image_path)
        print(f"GeoCalib -> fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
    else:
        if not all([args.fx, args.fy, args.cx, args.cy]):
            parser.error('Provide --fx/--fy/--cx/--cy or --geo_calib')
        fx, fy, cx, cy = args.fx, args.fy, args.cx, args.cy
    if not args.pts:
        parser.error('Need --pts or --pick')
    img = cv2.imread(args.image_path)
    if img is None:
        parser.error(f"Cannot load {args.image_path}")
    pts2d = np.array(args.pts, dtype=np.float64).reshape(4,2)
    K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])
    out = reconstruct_and_draw_stack(
        img, pts2d, args.cube_dim, args.num_stacks,
        args.num_per_stack, args.inner_spacing, K, None, args.draw_axes
    )
    cv2.imshow('Stack Reconstruction', out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
