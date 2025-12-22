import torch
import torch
from torch import nn
from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import load_image, rbd
from lightglue import viz2d
def simple_nms(scores, nms_radius: int):
    """ Fast Non-maximum suppression to remove nearby points """
    assert (nms_radius >= 0)

    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=nms_radius*2+1, stride=1, padding=nms_radius)

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)


def top_k_keypoints(keypoints, scores, k):
    if k >= len(keypoints):
        return keypoints, scores
    scores, indices = torch.topk(scores, k, dim=0, sorted=True)
    return keypoints[indices], scores


def sample_descriptors(keypoints, descriptors, s: int = 8):
    """ Interpolate descriptors at keypoint locations """
    b, c, h, w = descriptors.shape
    keypoints = keypoints - s / 2 + 0.5
    keypoints /= torch.tensor([(w*s - s/2 - 0.5), (h*s - s/2 - 0.5)],
                              ).to(keypoints)[None]
    keypoints = keypoints*2 - 1  # normalize to (-1, 1)
    args = {'align_corners': True} if torch.__version__ >= '1.3' else {}
    descriptors = torch.nn.functional.grid_sample(
        descriptors, keypoints.view(b, 1, -1, 2), mode='bilinear', **args)
    descriptors = torch.nn.functional.normalize(
        descriptors.reshape(b, c, -1), p=2, dim=1)
    return descriptors

class SuperPointPixo(SuperPoint):
    """SuperPoint Convolutional Detector and Descriptor

    SuperPoint: Self-Supervised Interest Point Detection and
    Description. Daniel DeTone, Tomasz Malisiewicz, and Andrew
    Rabinovich. In CVPRW, 2019. https://arxiv.org/abs/1712.07629

    """
    def __init__(self,**conf):
        super().__init__(**conf)

    def forward(self, data: dict) -> dict:
        """ Compute keypoints, scores, descriptors for image """
        for key in self.required_data_keys:
            assert key in data, f'Missing key {key} in data'
        image = data['image']
        if image.shape[1] == 3:  # RGB
            scale = image.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
            image = (image*scale).sum(1, keepdim=True)
        # Shared Encoder
        x = self.relu(self.conv1a(image))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))

        # Compute the dense keypoint scores
        cPa = self.relu(self.convPa(x))
        scores = self.convPb(cPa)
        scores = torch.nn.functional.softmax(scores, 1)[:, :-1]
        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h*8, w*8)
        scores = simple_nms(scores, self.conf['nms_radius'])

        # Discard keypoints near the image borders
        if self.conf['remove_borders']:
            pad = self.conf['remove_borders']
            scores[:, :pad] = -1
            scores[:, :, :pad] = -1
            scores[:, -pad:] = -1
            scores[:, :, -pad:] = -1

        # Extract keypoints
        best_kp = torch.where(scores > self.conf['detection_threshold'])
        scores = scores[best_kp]

        # Separate into batches
        keypoints = [torch.stack(best_kp[1:3], dim=-1)[best_kp[0] == i]
                     for i in range(b)]
        scores = [scores[best_kp[0] == i] for i in range(b)]

        # Keep the k keypoints with highest score
        if self.conf['max_num_keypoints'] is not None:
            keypoints, scores = list(zip(*[
                top_k_keypoints(k, s, self.conf['max_num_keypoints'])
                for k, s in zip(keypoints, scores)]))

        # Convert (h, w) to (x, y)
        keypoints = [torch.flip(k, [1]).float() for k in keypoints]

        # Compute the dense descriptors
        cDa = self.relu(self.convDa(x))
        descriptors = self.convDb(cDa)
        descriptors = torch.nn.functional.normalize(descriptors, p=2, dim=1)

        # Extract descriptors
        descriptors = [sample_descriptors(k[None], d[None], 8)[0]
                       for k, d in zip(keypoints, descriptors)]

        return {
            'keypoints': keypoints,
            'keypoint_scores': scores,
            'descriptors':descriptors,
        }

    def extract_batch(self, img: torch.Tensor, **conf) -> dict:
        """ Perform extraction with online resizing"""
        #if img.dim() == 3:
        #    img = img[None]  # add batch dim
        #assert img.dim() == 4 and img.shape[0] == 1
        #shape = img.shape[-2:][::-1]
        #img, scales = ImagePreprocessor(
        #    **{**self.preprocess_conf, **conf})(img)
        
        with torch.no_grad():
            feats = self.forward({'image': img.float()})
        torch.cuda.empty_cache()
        #feats['image_size'] = torch.tensor(shape)[None].to(img).float()
        #feats['keypoints'] = (feats['keypoints'] + .5) / scales[None] - .5
        return feats