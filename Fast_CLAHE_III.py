import imagej
import numpy as np
import os
import gc
import time
from tifffile import imwrite
from numba import njit, prange, set_num_threads, get_num_threads
from scyjava import jimport

# Force Java version if needed
# os.environ["JAVA_HOME"] = r"D:\Path\toFiji\java\win64\zulu21.42.19-ca-jdk21.0.7-win_x64"
# os.environ["PATH"] = os.path.join(os.environ["JAVA_HOME"], "bin") + ";" + os.environ["PATH"]

# Initialize FIJI/ImageJ 

fiji_path = r"D:\Path\to\Fiji"
ij = imagej.init(fiji_path, mode='interactive')

# Input image

image_path = r"C:\PAth\to\image.tif"
imp = ij.io().open(image_path)

# Convert Dataset -> ImagePlus if needed

ImagePlus = jimport("ij.ImagePlus")
if not isinstance(imp, ImagePlus):
    imp = ij.convert().convert(imp, ImagePlus)

# Helpers

def rgb_to_colorprocessor(arr):
    from scyjava import jimport
    ColorProcessor = jimport("ij.process.ColorProcessor")
    h, w, _ = arr.shape
    pixels = np.zeros(h * w, dtype=np.int32)
    r = arr[..., 0].flatten().astype(np.int32)
    g = arr[..., 1].flatten().astype(np.int32)
    b = arr[..., 2].flatten().astype(np.int32)
    pixels[:] = (r << 16) | (g << 8) | b
    return ColorProcessor(w, h, pixels)

def rgb_to_hsv_uint8(rgb):
    rgb = rgb.astype(np.float32)/255.0
    r, g, b = rgb[...,0], rgb[...,1], rgb[...,2]
    maxc, minc = np.maximum(np.maximum(r,g),b), np.minimum(np.minimum(r,g),b)
    v = maxc
    s = np.zeros_like(v)
    mask = maxc != 0
    s[mask] = (maxc[mask]-minc[mask])/maxc[mask]
    h = np.zeros_like(v)
    diff = maxc-minc + 1e-10
    mask_r = (maxc==r)&(diff>0)
    mask_g = (maxc==g)&(diff>0)
    mask_b = (maxc==b)&(diff>0)
    h[mask_r]=(g[mask_r]-b[mask_r])/diff[mask_r]
    h[mask_g]=2.0+(b[mask_g]-r[mask_g])/diff[mask_g]
    h[mask_b]=4.0+(r[mask_b]-g[mask_b])/diff[mask_b]
    h=(h/6.0)%1.0
    return np.stack([h*255, s*255, v*255], axis=-1).astype(np.uint8)

def hsv_to_rgb_uint8(hsv):
    h = hsv[...,0].astype(np.float32)/255.0
    s = hsv[...,1].astype(np.float32)/255.0
    v = hsv[...,2].astype(np.float32)/255.0
    i = np.floor(h*6).astype(np.int32)
    f = h*6 - i
    p = v*(1-s)
    q = v*(1-f*s)
    t = v*(1-(1-f)*s)
    r = np.zeros_like(h); g = np.zeros_like(h); b = np.zeros_like(h)
    i = i % 6
    r[i==0], g[i==0], b[i==0] = v[i==0], t[i==0], p[i==0]
    r[i==1], g[i==1], b[i==1] = q[i==1], v[i==1], p[i==1]
    r[i==2], g[i==2], b[i==2] = p[i==2], v[i==2], t[i==2]
    r[i==3], g[i==3], b[i==3] = p[i==3], q[i==3], v[i==3]
    r[i==4], g[i==4], b[i==4] = t[i==4], p[i==4], v[i==4]
    r[i==5], g[i==5], b[i==5] = v[i==5], p[i==5], q[i==5]
    rgb = np.stack([r,g,b], axis=-1)
    return np.clip(rgb*255, 0, 255).astype(np.uint8)

# Duplicate and convert to numpy

imp_copy = imp.duplicate()
arr = np.array(ij.py.from_java(imp_copy), dtype=np.uint8)
try: imp_copy.close()
except: pass
gc.collect()
System = jimport("java.lang.System")
System.gc()
print(f"Original shape: {arr.shape}, dtype: {arr.dtype}")

# CLAHE parameters nd RGB mode

blockRadius = 63
bins = 255
slope = 3.0
rgb_mode = "luminance"  # options: "luminance" or "per_channel"

# Numba CLAHE

@njit(parallel=True)
def clahe_fiji_exact(img, blockRadius=63, bins=255, slope=3.0):
    h, w = img.shape
    result = np.zeros_like(img, dtype=np.uint8)
    img32 = img.astype(np.float32)
    for y in prange(h):
        yMin = max(0, y-blockRadius)
        yMax = min(h, y+blockRadius+1)
        height = yMax - yMin
        hist = np.zeros(bins+1, dtype=np.int32)
        for yi in range(yMin,yMax):
            for xi in range(0, min(blockRadius+1, w)):
                v = int(img32[yi,xi]*bins/255.0+0.5)
                hist[v] += 1
        for x in range(w):
            xMin = max(0,x-blockRadius)
            xMax = min(w, x+blockRadius+1)
            width = xMax-xMin
            n = width*height
            limit = int(slope*n/bins + 0.5)
            if xMin>0:
                xi=xMin-1
                for yi in range(yMin,yMax):
                    v=int(img32[yi,xi]*bins/255.0+0.5)
                    hist[v]-=1
            if xMax<=w-1:
                xi=xMax-1
                for yi in range(yMin,yMax):
                    v=int(img32[yi,xi]*bins/255.0+0.5)
                    hist[v]+=1
            clippedHist = hist.copy()
            for _ in range(10):
                clippedEntries=0
                for i in range(bins+1):
                    if clippedHist[i]>limit:
                        clippedEntries += clippedHist[i]-limit
                        clippedHist[i]=limit
                if clippedEntries==0: break
                d = clippedEntries//(bins+1)
                m = clippedEntries%(bins+1)
                for i in range(bins+1): clippedHist[i]+=d
                for i in range(m): clippedHist[i]+=1
            hMin=0
            for i in range(bins+1):
                if clippedHist[i]!=0: hMin=i; break
            vpix = int(img32[y,x]*bins/255.0+0.5)
            cdf=sum(clippedHist[hMin:vpix+1])
            cdfMax=sum(clippedHist[hMin:])
            cdfMin=clippedHist[hMin]
            if cdfMax-cdfMin>0:
                val=int((cdf-cdfMin)/(cdfMax-cdfMin)*255+0.5)
            else:
                val=int(img32[y,x])
            result[y,x]=val
    return result

def clahe_fiji_fast(img, br, bins_, slope_):
    img = np.ascontiguousarray(img, dtype=np.uint8)
    if not hasattr(clahe_fiji_fast,"_compiled"):
        _ = clahe_fiji_exact(np.zeros((10,10), dtype=np.uint8), br, bins_, slope_)
        clahe_fiji_fast._compiled=True
    set_num_threads(get_num_threads())
    return clahe_fiji_exact(img, br, bins_, slope_)

def process_luminance(rgb):
    hsv = rgb_to_hsv_uint8(rgb)
    hsv[...,2] = clahe_fiji_fast(hsv[...,2], blockRadius, bins, slope)
    return hsv_to_rgb_uint8(hsv)

def process_rgb_channels(rgb):
    r = clahe_fiji_fast(rgb[...,0], blockRadius, bins, slope)
    g = clahe_fiji_fast(rgb[...,1], blockRadius, bins, slope)
    b = clahe_fiji_fast(rgb[...,2], blockRadius, bins, slope)
    return np.stack([r,g,b], axis=-1).astype(np.uint8)

# Apply CLAHE

start_time = time.time()
processed_stack = None
if arr.ndim==2:
    processed_stack = clahe_fiji_fast(arr, blockRadius, bins, slope)
elif arr.ndim==3 and arr.shape[2]==3:
    if rgb_mode=="luminance":
        processed_stack = process_luminance(arr)
    else:
        processed_stack = process_rgb_channels(arr)
elif arr.ndim==3:
    z,h,w = arr.shape
    processed_stack = np.zeros_like(arr, dtype=np.uint8)
    for i in range(z):
        processed_stack[i] = clahe_fiji_fast(arr[i], blockRadius, bins, slope)
        if (i+1)%5==0 or (i+1)==z: print(f"Processed slice {i+1}/{z}")
elif arr.ndim==4 and arr.shape[3]==3:
    z,h,w,_ = arr.shape
    processed_stack = np.zeros_like(arr, dtype=np.uint8)
    for i in range(z):
        if rgb_mode=="luminance":
            processed_stack[i] = process_luminance(arr[i])
        else:
            processed_stack[i] = process_rgb_channels(arr[i])
        if (i+1)%5==0 or (i+1)==z: print(f"Processed slice {i+1}/{z}")
else:
    raise RuntimeError("Unsupported image shape")
print(f"CLAHE done in {time.time()-start_time:.2f} s")

# Display in Fiji

ImagePlus = jimport("ij.ImagePlus")
if processed_stack.ndim==3 and processed_stack.shape[2]==3:
    cp = rgb_to_colorprocessor(processed_stack)
    imp_out = ImagePlus("CLAHE", cp)
else:
    imp_out = ij.py.to_java(processed_stack)
ij.ui().show(imp_out)

# Optional: save to same directory
# save_dir = os.path.dirname(image_path)
# save_name = os.path.splitext(os.path.basename(image_path))[0]
# out_path = os.path.join(save_dir, f"{save_name}_CLAHE.tif")
# imwrite(out_path, processed_stack)
# print(f"Processed image saved at: {out_path}")

input("Press Enter to exit...")

# Cleanup 

del arr, processed_stack
gc.collect()
System.gc()
