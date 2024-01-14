import numpy as np
from sklearn.linear_model import LinearRegression
import scipy
import open3d as o3d


def remove_outliers(point_cloud, threshold=1.0):
    # Calcola la media e la deviazione standard per ciascuna dimensione
    mean_values = np.mean(point_cloud, axis=0)
    std_deviation = np.std(point_cloud, axis=0)

    # Calcola la distanza euclidea di ciascun punto dalla media
    distances = np.linalg.norm(point_cloud - mean_values, axis=1)

    # Identifica gli indici degli outliers in base alla soglia specificata
    outlier_indices = np.where(distances > threshold * np.max(std_deviation))

    # Rimuovi gli outliers dalla point cloud
    filtered_point_cloud = np.delete(point_cloud, outlier_indices, axis=0)
    
    return filtered_point_cloud

def fit_plane_to_points(points):
    points = np.array(points)
    points = remove_outliers(points)

    # Salva i points filtrati in un np array e ritornalo in formato open3d pointcloud
    #o3d_points = o3d.geometry.PointCloud()
    #o3d_points.points = o3d.utility.Vector3dVector(points)

    ones_column = np.ones((points.shape[0], 1))
    # Keep only first 2 dimensions of points
    points_extended = np.hstack([points[:, :2], ones_column])
    print(points_extended.shape)
    coeff, _, _, _ = np.linalg.lstsq(points_extended, points[:, -1], rcond=None)
    A, B, D = coeff
    C = -1
    #est_d = (- A * points[:, 0] - B * points[:, 1] - C * points[:, 2])
    #D = np.mean(est_d)

    return A, B, C, D #, o3d_points


def fit_line(points):
    # Costruisci la matrice A e il vettore B
    A = np.column_stack((points, np.ones(len(points))))
    B = -np.ones(len(points))

    # Risolvi il sistema Ax = B mediante minimi quadrati
    coeff, _, _, _ = np.linalg.lstsq(A, B, rcond=None)

    # Estrai i coefficienti
    a, b, c, d = coeff

    return a, b, c, d

def get_rotation_between_lines(v1, v2) -> np.ndarray((3,3)):
    # Passo 1
    k = np.array([0, 0, 1])  # Asse z
    n1 = np.cross(v1, k)
    n2 = np.cross(v2, k)

    # Passo 2
    n1_norm = n1 / np.linalg.norm(n1)
    n2_norm = n2 / np.linalg.norm(n2)

    # Passo 3
    theta = np.arccos(np.dot(n1_norm, n2_norm))

    # Passo 4
    R = np.array([[np.cos(theta), -np.sin(theta), 0],
                  [np.sin(theta), np.cos(theta), 0],
                  [0, 0, 1]])

    return R

def find_intersection(a1, b1, c1, d1, a2, b2, c2, d2) -> np.ndarray(3):
    # Coefficients of the plane equation
    A = a1 - a2
    B = b1 - b2
    C = c1 - c2
    D = d1 - d2

    # Choose one variable (e.g., z) and set it to 0
    z = 0

    # Solve for x and y
    x = -D / A
    y = -D / B

    # Return the intersection point
    return np.array([x, y, z])


def compute_rotation(p1, p2):
    A1, B1, C1, D1 = p1
    A2, B2, C2, D2 = p2
    Z =  (-A1 * 1 - B1 * 1 - D1) / C1
    point = np.array([1, 1, Z])

    # Normalizza i vettori normali
    n1 = np.array([A1, B1, C1])
    n2 = np.array([A2, B2, C2])
    n1 /= np.linalg.norm(n1)
    n2 /= np.linalg.norm(n2)

    # Trova l'asse di rotazione
    asse_rotazione = np.cross(n1, n2)
    asse_rotazione /= np.linalg.norm(asse_rotazione)
    print(asse_rotazione)
    print(np.sum(asse_rotazione**2))
    # Trova l'angolo di rotazione
    angolo_rotazione = np.arccos(np.dot(n1, n2))
    # Matrice di rotazione
    R = np.array([
        [np.cos(angolo_rotazione) + asse_rotazione[0]**2 * (1 - np.cos(angolo_rotazione)),
         asse_rotazione[0] * asse_rotazione[1] * (1 - np.cos(angolo_rotazione)) - asse_rotazione[2] * np.sin(angolo_rotazione),
         asse_rotazione[0] * asse_rotazione[2] * (1 - np.cos(angolo_rotazione)) + asse_rotazione[1] * np.sin(angolo_rotazione)],
        [asse_rotazione[1] * asse_rotazione[0] * (1 - np.cos(angolo_rotazione)) + asse_rotazione[2] * np.sin(angolo_rotazione),
         np.cos(angolo_rotazione) + asse_rotazione[1]**2 * (1 - np.cos(angolo_rotazione)),
         asse_rotazione[1] * asse_rotazione[2] * (1 - np.cos(angolo_rotazione)) - asse_rotazione[0] * np.sin(angolo_rotazione)],
        [asse_rotazione[2] * asse_rotazione[0] * (1 - np.cos(angolo_rotazione)) - asse_rotazione[1] * np.sin(angolo_rotazione),
         asse_rotazione[2] * asse_rotazione[1] * (1 - np.cos(angolo_rotazione)) + asse_rotazione[0] * np.sin(angolo_rotazione),
         np.cos(angolo_rotazione) + asse_rotazione[2]**2 * (1 - np.cos(angolo_rotazione))]
    ])

    p1 = rotate_plane_equation(p1, R, point)
    # Matrice di rototraslazione
    #t_vector = compute_translation_vector(p1, p2)
    #print(t_vector)

    return R, p1 #, t_vector


def rotate_plane_equation(coefficients_plane, rotation_matrix, point):
    # Estrai i coefficienti del piano originale
    a, b, c, d = coefficients_plane
    
    # Costruisci il vettore normale al piano
    normal_vector = np.array([a, b, c])

    # Applica la rotazione al vettore normale
    rotated_normal_vector = np.dot(rotation_matrix, normal_vector)
    
    # Estrai i nuovi coefficienti dopo la rotazione
    a_rotated, b_rotated, c_rotated = rotated_normal_vector

    # Ruota il punto
    p_rot = np.dot(rotation_matrix, point)
    d_rotated = -a_rotated * p_rot[0] - b_rotated * p_rot[1] - c_rotated * p_rot[2]
    print(f'point: {point}, p_rot: {p_rot}')
    print(f'coefficients: a_rotated: {a_rotated}, b_rotated: {b_rotated}, c_rotated: {c_rotated}, d_rotated: {d_rotated}')
    print(f'coefficients old a: {a}, b: {b}, c: {c}, d: {d}')
    return a_rotated, b_rotated, c_rotated, d_rotated



def get_plane_pcd(coeffs, color: list = [0, 0, 1]):
    # Estrai i coefficienti
    A, B, C, D = coeffs

    # Calcola i punti del piano
    x = np.linspace(-1, 1, 300)
    y = np.linspace(-1, 1, 300)
    X, Y = np.meshgrid(x, y)
    Z = (- A * X - B * Y - D) / C
    # Converti in pointcloud
    points = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))

    # Create Open3D PointCloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.paint_uniform_color(color)
    return point_cloud

def translation_vector(p1, p2):
    # Estrai i coefficienti dei piani
    a1, b1, c1, d1 = p1
    a2, b2, c2, d2 = p2
    n1 = np.array([a1, b1, c1])
    n2 = np.array([a2, b2, c2])
    
    # Trovo un punto appartenente a p1 e p2
    x, y = 0, 0
    z = (-a1*x - b1*y - d1)/c1
    point_on_plane1 = np.array([x, y, z])
    z = (-a2*x - b2*y - d2)/c2
    point_on_plane2 = np.array([x, y, z])

    vector_to_point =  point_on_plane2 - point_on_plane1
    
    # scale_factor = -np.dot(vector_to_point, n2) / np.linalg.norm(n2)**2
    
    # # Calcola la proiezione del punto sul piano
    # projected_point = point_on_plane2 + scale_factor * n2
    # print(f'point_on_plane1: {point_on_plane1}, point_on_plane2: {point_on_plane2}, projected_point: {projected_point}')
    # # Calcolo il vettore di traslazione
    # t_vector = projected_point - point_on_plane1

    return vector_to_point

def distance_between_planes(coefficients_plane1, coefficients_plane2):
    # Scegli un punto su uno dei piani (ad esempio, l'origine)
    

    # Estrai i coefficienti dei piani
    a1, b1, c1, d1 = coefficients_plane1
    a2, b2, c2, d2 = coefficients_plane2

    x = 0 
    y = 0
    z = (-a1*x - b1*y - d1)/c1
    point_on_plane1 = np.array([x, y, z])

    # Calcola la distanza
    distance = np.abs(a2 * point_on_plane1[0] + b2 * point_on_plane1[1] + c2 * point_on_plane1[2] + d2) / np.sqrt(a2**2 + b2**2 + c2**2)

    return distance
