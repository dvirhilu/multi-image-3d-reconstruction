def cyl_2_cart(r, phi, z):
    x = r*np.cos(phi)
    y = r*np.sin(phi)
    return (x, y, z)

def cart_2_cyl(x, y, z):
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return (r, phi, z)

def pol_2_cart2d(r, phi):
    x = r*np.cos(phi)
    y = r*np.sin(phi)
    return (x, y)

def component_in_dir(vector, direction):
    return np.dot(vector, direction)/norm(direction)

def proj_2_vec(orig_vector, proj_direction):
    return component_in_dir(orig_vector, proj_direction) * unit_vect(proj_direction)

def proj_2_plane(vector, plane_normal):
    return vector - proj_2_vec(vector, plane_normal)

def proj_2_xy(vector):
    proj = proj_2_plane(vector, np.array([0, 0, 1]))
    return np.array([proj[0], proj[1]])

def get_vec_sin_angle(vec1, vec2):
    return cross_2d(vec1, vec2) / (norm(vec1)*norm(vec2))

def skew_sym(a):
    return np.array([
        [0,     -a[2],  a[1]],
        [a[2],  0,      -a[0]],
        [-a[1], a[0],   0]
    ])