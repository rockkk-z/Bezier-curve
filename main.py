import taichi as ti
import numpy as np

ti.init(arch=ti.gpu)

WIDTH = 800
HEIGHT = 800
MAX_CONTROL_POINTS = 100
NUM_SEGMENTS = 1000

# ===================== GPU缓冲区 =====================
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))
gui_points = ti.Vector.field(2, dtype=ti.f32, shape=MAX_CONTROL_POINTS)
gui_indices = ti.field(dtype=ti.i32, shape=MAX_CONTROL_POINTS * 2)
curve_points_field = ti.Vector.field(2, dtype=ti.f32, shape=NUM_SEGMENTS + 1)

# ===================== De Casteljau =====================
def de_casteljau(points, t):
    if len(points) == 1:
        return points[0]
    next_points = []
    for i in range(len(points) - 1):
        p0 = points[i]
        p1 = points[i + 1]
        x = (1 - t) * p0[0] + t * p1[0]
        y = (1 - t) * p0[1] + t * p1[1]
        next_points.append([x, y])
    return de_casteljau(next_points, t)

# ===================== B样条 =====================
def bspline_curve(points):
    if len(points) < 4:
        return []

    result = []

    M = (1/6) * np.array([
        [-1, 3, -3, 1],
        [3, -6, 3, 0],
        [-3, 0, 3, 0],
        [1, 4, 1, 0]
    ], dtype=np.float32)

    seg_num = len(points) - 3
    samples_per_seg = max(1, NUM_SEGMENTS // seg_num)

    for i in range(seg_num):
        P = np.array(points[i:i+4], dtype=np.float32)

        for j in range(samples_per_seg):
            t = j / samples_per_seg
            T = np.array([t**3, t**2, t, 1], dtype=np.float32)
            pt = T @ M @ P
            result.append(pt)

    return result

# ===================== GPU Kernel =====================
@ti.kernel
def clear_pixels():
    for i, j in pixels:
        pixels[i, j] = ti.Vector([0.0, 0.0, 0.0])

@ti.kernel
def draw_curve_antialiasing(n: ti.i32):
    for i in range(n):
        x, y = curve_points_field[i]

        fx = x * WIDTH
        fy = y * HEIGHT

        for dx in ti.static(range(-1, 2)):
            for dy in ti.static(range(-1, 2)):
                px = int(fx + dx)
                py = int(fy + dy)

                if 0 <= px < WIDTH and 0 <= py < HEIGHT:
                    dist = ti.sqrt((fx - (px + 0.5))**2 +
                                   (fy - (py + 0.5))**2)

                    weight = ti.max(0.0, 1.0 - dist / 1.5)

                    # 防止颜色叠加过亮
                    new_color = pixels[px, py] + ti.Vector([0.0, weight, 0.0])
                    pixels[px, py] = ti.min(new_color, ti.Vector([0.0, 1.0, 0.0]))

# ===================== 主程序 =====================
def main():
    window = ti.ui.Window("Bezier & B-Spline", (WIDTH, HEIGHT))
    canvas = window.get_canvas()

    control_points = []
    use_bspline = False

    while window.running:
        # ===== 事件 =====
        for e in window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.LMB:
                if len(control_points) < MAX_CONTROL_POINTS:
                    pos = window.get_cursor_pos()
                    control_points.append(pos)

            elif e.key == 'c':
                control_points = []

            elif e.key == 'b':
                use_bspline = not use_bspline
                print("B-spline mode" if use_bspline else "Bezier mode")

        clear_pixels()

        current_count = len(control_points)

        # ===== 曲线计算 =====
        if current_count >= 2:
            if use_bspline and current_count >= 4:
                curve_points_np = bspline_curve(control_points)
                curve_points_np = np.array(curve_points_np, dtype=np.float32)
            else:
                curve_points_np = np.zeros((NUM_SEGMENTS + 1, 2), dtype=np.float32)
                for t_int in range(NUM_SEGMENTS + 1):
                    t = t_int / NUM_SEGMENTS
                    curve_points_np[t_int] = de_casteljau(control_points, t)

            if len(curve_points_np) > 0:
                curve_points_field.from_numpy(curve_points_np)
                draw_curve_antialiasing(len(curve_points_np))

        canvas.set_image(pixels)

        # ===== 控制点绘制（对象池）=====
        if current_count > 0:
            np_points = np.full((MAX_CONTROL_POINTS, 2), -10.0, dtype=np.float32)
            np_points[:current_count] = np.array(control_points, dtype=np.float32)
            gui_points.from_numpy(np_points)

            canvas.circles(gui_points, radius=0.006, color=(1, 0, 0))

            if current_count >= 2:
                np_indices = np.zeros(MAX_CONTROL_POINTS * 2, dtype=np.int32)
                indices = []
                for i in range(current_count - 1):
                    indices.extend([i, i + 1])
                np_indices[:len(indices)] = np.array(indices, dtype=np.int32)
                gui_indices.from_numpy(np_indices)

                canvas.lines(gui_points,
                             width=0.002,
                             indices=gui_indices,
                             color=(0.5, 0.5, 0.5))

        window.show()

if __name__ == '__main__':
    main()