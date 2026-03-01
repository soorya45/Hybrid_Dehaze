import cv2
import numpy as np

# ================= NIGHT MODEL =================
class NightModel:
    def predict(self, img):

        img = cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)

        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l,a,b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
        l = clahe.apply(l)

        enhanced = cv2.merge((l,a,b))
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

        enhanced = cv2.fastNlMeansDenoisingColored(enhanced,None,10,10,7,21)

        return enhanced


# ================= DENSE MODEL =================
class DenseModel:
    def predict(self, img):

        def dark_channel(im, k=15):
            return cv2.erode(np.min(im,2), np.ones((k,k)))

        def atm_light(im, dark):
            h,w = dark.shape
            n = int(max(h*w*0.001,1))
            idx = np.argsort(dark.ravel())[-n:]
            return np.max(im.reshape(-1,3)[idx], axis=0)

        def transmission(im,A,omega=0.98):
            return 1 - omega * dark_channel(im/A,15)

        def recover(im,t,A):
            t = np.clip(t,0.08,1)
            return (im-A)/t[:,:,None] + A

        im = img.astype(np.float64)/255

        dark = dark_channel(im)
        A = atm_light(im,dark)
        t = transmission(im,A)
        clear = recover(im,t,A)
        clear = np.clip(clear*255,0,255).astype(np.uint8)

        return clear


# ================= LIGHT MODEL =================
class LightModel:
    def predict(self, img):

        def dark_channel(img, size=15):
            b, g, r = cv2.split(img)
            min_img = cv2.min(cv2.min(r, g), b)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
            dark = cv2.erode(min_img, kernel)
            return dark

        def atmospheric_light(img, dark):
            h, w = dark.shape
            num_pixels = h * w
            num_bright = max(int(num_pixels * 0.001), 1)

            dark_vec = dark.reshape(num_pixels)
            img_vec = img.reshape(num_pixels, 3)

            indices = dark_vec.argsort()[-num_bright:]
            A = np.mean(img_vec[indices], axis=0)
            return A

        def transmission_estimate(img, A, omega=0.95, size=15):
            norm_img = img / A
            dark = dark_channel(norm_img, size)
            return 1 - omega * dark

        def recover(img, t, A, t0=0.1):
            t = np.maximum(t, t0)
            J = (img - A) / t[:, :, None] + A
            return np.clip(J, 0, 255).astype(np.uint8)

        def simple_white_balance(img):
            result = img.copy().astype(np.float32)
            avg_b = np.mean(result[:,:,0])
            avg_g = np.mean(result[:,:,1])
            avg_r = np.mean(result[:,:,2])

            avg_gray = (avg_b + avg_g + avg_r) / 3

            result[:,:,0] *= (avg_gray / avg_b)
            result[:,:,1] *= (avg_gray / avg_g)
            result[:,:,2] *= (avg_gray / avg_r)

            return np.clip(result, 0, 255).astype(np.uint8)

        def enhance_contrast(img):
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)

            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply(l)

            lab = cv2.merge((l, a, b))
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        def gamma_correction(img, gamma=0.85):
            inv = 1.0 / gamma
            table = np.array([(i / 255.0) ** inv * 255 for i in range(256)]).astype("uint8")
            return cv2.LUT(img, table)

        img_float = img.astype(np.float64)

        dark = dark_channel(img_float)
        A = atmospheric_light(img_float, dark)
        t = transmission_estimate(img_float, A)
        dehazed = recover(img_float, t, A)

        dehazed = enhance_contrast(dehazed)
        dehazed = gamma_correction(dehazed, gamma=0.85)
        dehazed = simple_white_balance(dehazed)

        return dehazed