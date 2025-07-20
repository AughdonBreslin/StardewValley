import cv2 as cv
import numpy as np
import win32con, win32gui, win32ui


class WindowCapture:
    def __init__(self, window_name=None):
        if window_name is None:
            self.hwnd = win32gui.GetDesktopWindow()
        else:
            self.hwnd = win32gui.FindWindow(None, window_name)
        if not self.hwnd:
            raise Exception(f"Window '{window_name}' not found. Make sure the window is open. Current open windows: {self.list_windows()}")
        window_rect = win32gui.GetWindowRect(self.hwnd)
        title_bar_height = win32con.SM_CYSIZE
        border_width = win32con.SM_CXBORDER
        self.w = window_rect[2] - window_rect[0] - 2 * border_width
        self.h = window_rect[3] - window_rect[1] - title_bar_height - border_width
        self.cropped_x = border_width
        self.cropped_y = title_bar_height + border_width

    def get_screenshot(self):
        self.hwnd = win32gui.FindWindow(None, 'Stardew Valley')
        windowDC = win32gui.GetWindowDC(self.hwnd)
        dcObject = win32ui.CreateDCFromHandle(windowDC)
        compatibleDC = dcObject.CreateCompatibleDC()
        dataBitmap = win32ui.CreateBitmap()
        dataBitmap.CreateCompatibleBitmap(dcObject, self.w, self.h)
        compatibleDC.SelectObject(dataBitmap)
        compatibleDC.BitBlt((0, 0), (self.w, self.h), dcObject, (self.cropped_x, self.cropped_y), win32con.SRCCOPY)

        signedIntsArray = dataBitmap.GetBitmapBits(True)
        img = np.frombuffer(signedIntsArray, dtype='uint8')
        img.shape = (self.h, self.w, 4)
        img = cv.cvtColor(img, cv.COLOR_BGRA2BGR)  # Convert from BGRA to BGR

        # Clean up resources
        dcObject.DeleteDC()
        compatibleDC.DeleteDC()
        win32gui.ReleaseDC(self.hwnd, windowDC)
        win32gui.DeleteObject(dataBitmap.GetHandle())

        return img
    
    @staticmethod
    def list_windows():
        def enum_handler(hwnd, windows):
            if win32gui.IsWindowVisible(hwnd) and win32gui.IsWindowEnabled(hwnd):
                windows.append((hwnd, win32gui.GetWindowText(hwnd)))
        windows = []
        win32gui.EnumWindows(enum_handler, windows)
        return windows


