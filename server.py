from flask import Flask, request, send_file, render_template 
import numpy as np 
from plyfile import PlyData 
import torch 
from scipy.spatial.transform import Rotation as R 
from gsplat import rasterization 
from PIL import Image 
import io 

app = Flask(__name__, static_folder='static', template_folder='templates') 

def load_gs_ply(ply_path): 
    plydata = PlyData.read(ply_path) 
    vertex = plydata['vertex'] 
    means = np.stack((vertex['x'], vertex['y'], vertex['z']), axis=-1).astype(np.float32) 
    scales = np.exp(np.stack((vertex['scale_0'], vertex['scale_1'], vertex['scale_2']), axis=-1)).astype(np.float32) 
    quats = np.stack((vertex['rot_0'], vertex['rot_1'], vertex['rot_2'], vertex['rot_3']), axis=-1).astype(np.float32) 
    quats /= np.linalg.norm(quats, axis=-1, keepdims=True) 
    opacities = 1 / (1 + np.exp(-vertex['opacity'])).astype(np.float32) 
    shs = np.zeros((means.shape[0], 16, 3), dtype=np.float32) 
    shs[:, 0, 0] = vertex['f_dc_0'] 
    shs[:, 0, 1] = vertex['f_dc_1'] 
    shs[:, 0, 2] = vertex['f_dc_2'] 
    for i in range(45): 
        shs[:, (i // 3) + 1, i % 3] = vertex[f'f_rest_{i}'] 
    return means, quats, scales, opacities, shs 


def create_viewmat(azimuth_deg, elevation_deg, x, y, z): 
    rot = R.from_euler('xyz', [elevation_deg, azimuth_deg, 0], degrees=True).as_matrix() 
    trans = np.array([x, y, z]) 
    c2w = np.eye(4) 
    c2w[:3, :3] = rot 
    c2w[:3, 3] = trans 
    w2c = np.linalg.inv(c2w) 
    return torch.tensor(w2c, dtype=torch.float32) 


# Load Model Once 
# 
means_np, quats_np, scales_np, opacities_np, shs_np = load_gs_ply('static/models/model_high.ply') 
means = torch.from_numpy(means_np).cuda() 
quats = torch.from_numpy(quats_np).cuda() 
scales = torch.from_numpy(scales_np).cuda() 
opacities = torch.from_numpy(opacities_np).cuda().squeeze(-1) 
shs = torch.from_numpy(shs_np).cuda() 

# Performance upgrade 
del means_np, quats_np, scales_np, opacities_np, shs_np 


def rendering(azimuth_deg, elevation_deg, x, y, z, fx, fy, cx, cy, width, height): 
    print(f"GPU memory before: {torch.cuda.memory_allocated() / 1024**3:.2f} GB") 
    viewmat = create_viewmat(azimuth_deg, elevation_deg, x, y, z).cuda().unsqueeze(0) 
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32) 
    K_batch = K.cuda().unsqueeze(0) 
    colors_rendered, alphas, _ = rasterization( 
                                               means=means, 
                                               quats=quats, 
                                               scales=scales, 
                                               opacities=opacities, 
                                               colors=shs, 
                                               viewmats=viewmat, 
                                               Ks=K_batch, 
                                               width=int(width), 
                                               height=int(height), 
                                               packed=False, 
                                               sh_degree=0, 
                                               backgrounds=None, 
                                               render_mode='RGB' 
                                               ) 
    img = colors_rendered[0].cpu().numpy() 
    img = np.clip(img, 0, 1) * 255 
    img = img.astype(np.uint8) 
    pil_img = Image.fromarray(img) 
    img_byte_arr = io.BytesIO() 
    pil_img.save(img_byte_arr, format='JPEG', quality=70) 
    img_byte_arr.seek(0) 
    print(f"GPU memory after: {torch.cuda.memory_allocated() / 1024**3:.2f} GB") 
    return img_byte_arr 


@app.route('/') 
def home(): 
    return render_template('index.html') 

@app.route('/render', methods=['POST']) 
def render(): 
    data = request.json 
    azimuth = float(data.get('angle', 180)) 
    elevation = float(data.get('elevation', 0)) 
    x = float(data.get('x', 0)) 
    y = float(data.get('y', 0)) 
    z = float(data.get('z', 5.0)) 
    fx = float(data.get('fx', 1300.0)) 
    fy = float(data.get('fy', 800.0)) 
    cx = float(data.get('cx', 400.0)) 
    cy = float(data.get('cy', 300.0)) 
    width = float(data.get('width', 800)) 
    height = float(data.get('height', 600)) 
    img_byte_arr = rendering(azimuth, elevation, x, y, z, fx, fy, cx, cy, width, height) 
    return send_file(img_byte_arr, mimetype='image/jpeg') 

if __name__ == '__main__': 
    app.run(host='0.0.0.0', port=8000, debug=False)
