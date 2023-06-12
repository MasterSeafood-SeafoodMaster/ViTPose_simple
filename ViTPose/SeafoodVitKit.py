import torch
from torch import Tensor

import cv2
import numpy as np

from PIL import Image
from torchvision.transforms import transforms

from ViTPose.models.model import ViTPose
from utils.visualization import draw_points_and_skeleton, joints_dict
from utils.dist_util import get_dist_info, init_dist
from utils.top_down_eval import keypoints_from_heatmaps


 
@torch.no_grad()
class VitModel:
	def __init__(self, modelPath):
		from configs.ViTPose_base_coco_256x192 import model as model_cfg
		from configs.ViTPose_base_coco_256x192 import data_cfg

		self.modelPath = modelPath
		self.img_size = data_cfg['image_size']

		if torch.cuda.is_available():
			self.device=torch.device("cuda") 
		else:
			self.device=torch.device('cpu')

		self.vit_pose = ViTPose(model_cfg)

		# Prepare model
		self.ckpt = torch.load(self.modelPath)

		if 'state_dict' in self.ckpt:
			self.vit_pose.load_state_dict(self.ckpt['state_dict'])
		else:
			self.vit_pose.load_state_dict(self.ckpt)
		self.vit_pose.to(self.device)
		print(f">>> Model loaded: {self.modelPath}")

	def vitPred(self, image):
		self.img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		#self.img = image[:,:,::-1]
		self.img = Image.fromarray(self.img)
		self.org_w, self.org_h = self.img.size

		self.img_tensor = transforms.Compose (
			[transforms.Resize((self.img_size[1], self.img_size[0])),
			 transforms.ToTensor()]
		)(self.img).unsqueeze(0).to(self.device)

		self.heatmaps = self.vit_pose(self.img_tensor).detach().cpu().numpy()
		
		self.points, self.prob = keypoints_from_heatmaps(heatmaps=self.heatmaps, center=np.array([[self.org_w//2, self.org_h//2]]), scale=np.array([[self.org_w, self.org_h]]),
											   unbiased=True, use_udp=True)

		self.points = np.concatenate([self.points[:, :, ::-1], self.prob], axis=2)

		return self.points[0]

	def visualization(self, image, points):
		self.points = points
		self.img = image
		point = self.points
		#self.img = np.array(self.img)[:, :, ::-1] # RGB to BGR for cv2 modules
		self.img = draw_points_and_skeleton(self.img.copy(), point, joints_dict()['coco']['skeleton'], person_index=0,
									   points_color_palette='gist_rainbow', skeleton_color_palette='jet',
									   points_palette_samples=10, confidence_threshold=0.4)

		return self.img

	def local2world(self, points, yolobox):
		self.points = points
		self.yolobox = np.array([yolobox[1], yolobox[0]])

		self.world=self.points[:, 0:2]+self.yolobox
		self.world = self.world.astype(int)
		return np.flip(self.world, axis=1)

	
"""
CKPT_PATH = "./vitpose-b-multi-coco.pth"

frame = cv2.imread('./00001_38_0.jpg')
vit_model = VitModel(CKPT_PATH)
points = vit_model.vitPred(frame)

frame = vit_model.visualization(frame, points)
cv2.imshow("l", frame)
cv2.waitKey(0)
"""
