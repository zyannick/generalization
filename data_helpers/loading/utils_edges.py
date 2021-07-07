import numpy as np
import cv2

def transform_img_sobel(img, blur_kernel, operator_kernel, distance_transform=0):
    scale = 1
    delta = 0
    ddepth = cv2.CV_16S

    if blur_kernel > 0:
        src = cv2.GaussianBlur(img, (blur_kernel, blur_kernel), 0)
    else:
        src = img

    # print(src.shape)

    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=operator_kernel, scale=scale, delta=delta,
                       borderType=cv2.BORDER_DEFAULT)
    # Gradient-Y
    # grad_y = cv.Scharr(gray,ddepth,0,1)
    grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=operator_kernel, scale=scale, delta=delta,
                       borderType=cv2.BORDER_DEFAULT)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    #out = cv2.distanceTransform(overlay, cv2.DIST_L2, 5)

    grad = cv2.cvtColor(grad, cv2.COLOR_GRAY2BGR)

    return grad


def kirsch_filter_all_directions(gray):
    if gray.ndim > 2:
        raise Exception("illegal argument: input must be a single channel image (gray)")
    kernelG1 = np.array([[5, 5, 5],
                         [-3, 0, -3],
                         [-3, -3, -3]], dtype=np.float32)
    kernelG2 = np.array([[5, 5, -3],
                         [5, 0, -3],
                         [-3, -3, -3]], dtype=np.float32)
    kernelG3 = np.array([[5, -3, -3],
                         [5, 0, -3],
                         [5, -3, -3]], dtype=np.float32)
    kernelG4 = np.array([[-3, -3, -3],
                         [5, 0, -3],
                         [5, 5, -3]], dtype=np.float32)
    kernelG5 = np.array([[-3, -3, -3],
                         [-3, 0, -3],
                         [5, 5, 5]], dtype=np.float32)
    kernelG6 = np.array([[-3, -3, -3],
                         [-3, 0, 5],
                         [-3, 5, 5]], dtype=np.float32)
    kernelG7 = np.array([[-3, -3, 5],
                         [-3, 0, 5],
                         [-3, -3, 5]], dtype=np.float32)
    kernelG8 = np.array([[-3, 5, 5],
                         [-3, 0, 5],
                         [-3, -3, -3]], dtype=np.float32)

    g1 = cv2.normalize(cv2.filter2D(gray, cv2.CV_32F, kernelG1), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    g2 = cv2.normalize(cv2.filter2D(gray, cv2.CV_32F, kernelG2), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    g3 = cv2.normalize(cv2.filter2D(gray, cv2.CV_32F, kernelG3), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    g4 = cv2.normalize(cv2.filter2D(gray, cv2.CV_32F, kernelG4), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    g5 = cv2.normalize(cv2.filter2D(gray, cv2.CV_32F, kernelG5), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    g6 = cv2.normalize(cv2.filter2D(gray, cv2.CV_32F, kernelG6), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    g7 = cv2.normalize(cv2.filter2D(gray, cv2.CV_32F, kernelG7), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    g8 = cv2.normalize(cv2.filter2D(gray, cv2.CV_32F, kernelG8), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    magn = cv2.max(
        g1, cv2.max(
            g2, cv2.max(
                g3, cv2.max(
                    g4, cv2.max(
                        g5, cv2.max(
                            g6, cv2.max(
                                g7, g8
                            )
                        )
                    )
                )
            )
        )
    )
    return magn


def kirsch_filter(gray):
    kernelG1 = np.array([[5, 5, 5],
                         [-3, 0, -3],
                         [-3, -3, -3]], dtype=np.float32)
    kernelG3 = np.array([[5, -3, -3],
                         [5, 0, -3],
                         [5, -3, -3]], dtype=np.float32)

    grad_x = cv2.filter2D(gray, cv2.CV_32F, kernelG1)
    grad_y = cv2.filter2D(gray, cv2.CV_32F, kernelG3)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    grad_kirsh = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    return grad_kirsh



def treble_edges(img, blur_kernel, operator_kernel):
    ddepth = cv2.CV_16S
    scale = 1
    delta = 0

    if blur_kernel > 0:
        src = cv2.GaussianBlur(img, (blur_kernel, blur_kernel), 0)
    else:
        src = img

    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape

    dst_laplacien = cv2.Laplacian(gray, ddepth, ksize=operator_kernel)
    grad_laplacien = cv2.convertScaleAbs(dst_laplacien)
    grad_laplacien = np.reshape(grad_laplacien, (h, w, 1))

    grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=operator_kernel, scale=scale, delta=delta,
                       borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=operator_kernel, scale=scale, delta=delta,
                       borderType=cv2.BORDER_DEFAULT)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    grad_sobel = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    grad_sobel = np.reshape(grad_sobel, (h, w, 1))

    grad_kirsh = kirsch_filter(gray)
    grad_kirsh = np.reshape(grad_kirsh, (h, w, 1))

    tmp = np.concatenate((grad_laplacien, grad_sobel), axis=2)

    gradient_results = np.concatenate((tmp, grad_kirsh), axis=2)

    return gradient_results


def laplace_dev(img, blur_kernel, operator_kernel):
    ddepth = cv2.CV_16S

    if blur_kernel > 0:
        src = cv2.GaussianBlur(img, (blur_kernel, blur_kernel), 0)
    else:
        src = img

    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    dst = cv2.Laplacian(gray, ddepth, ksize=operator_kernel)
    # [laplacian]
    # [convert]
    # converting back to uint8
    grad = cv2.convertScaleAbs(dst)

    grad = cv2.cvtColor(grad, cv2.COLOR_GRAY2BGR)

    return grad
