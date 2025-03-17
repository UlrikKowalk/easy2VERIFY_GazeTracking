from GazeData import GazeData
import cv2
from torch.utils.data import DataLoader


library = GazeData(directory="D:/easy2VERIFY_Dataset", device='cpu')

image, target, head_rotation, head_elevation, head_tilt = library[100]
print(target, head_rotation, head_elevation, head_tilt)
cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# data_loader = DataLoader(dataset=library,
#                                   batch_size=1,
#                                   num_workers=1,
#                                   persistent_workers=True,
#                                   shuffle=False)
#
#
# for image, target, head_rotation, head_elevation, head_tilt in data_loader.dataset[0]:

    # print(target, head_rotation, head_elevation, head_tilt)
    # cv2.imshow(image)

