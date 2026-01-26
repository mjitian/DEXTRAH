import numpy as np
from scipy.spatial.transform import Rotation as R

def compare_quat(q1, q2_target):
    """
    Compares two quaternions q1 and q2_target.
    Returns the angular difference in degrees.
    Handles the fact that q and -q represent the same rotation.
    """
    # q1 is from matrix (scipy format: x, y, z, w)
    
    # Check assuming target is [x, y, z, w]
    dot_xyzw = np.abs(np.dot(q1, q2_target))
    dot_xyzw = np.clip(dot_xyzw, -1.0, 1.0)
    angle_xyzw = 2 * np.arccos(dot_xyzw)
    deg_xyzw = np.degrees(angle_xyzw)

    # Check assuming target is [w, x, y, z] -> Convert to [x, y, z, w] for comparison
    q2_reordered = np.array([q2_target[1], q2_target[2], q2_target[3], q2_target[0]])
    dot_wxyz = np.abs(np.dot(q1, q2_reordered))
    dot_wxyz = np.clip(dot_wxyz, -1.0, 1.0)
    angle_wxyz = 2 * np.arccos(dot_wxyz)
    deg_wxyz = np.degrees(angle_wxyz)

    return deg_xyzw, deg_wxyz

def analyze_pose(name, matrix, target_quat):
    print(f"==================================================")
    print(f" ANALYZING: {name}")
    print(f"==================================================")
    print("Transformation Matrix:")
    print(matrix)
    print(f"Target Quaternion (Provided): {target_quat}")
    
    rot_matrix = matrix[:3, :3]
    translation = matrix[:3, 3]
    
    # Create rotation object from matrix
    r = R.from_matrix(rot_matrix)
    calculated_quat = r.as_quat() # returns (x, y, z, w)
    
    print(f"Calculated Quaternion (from matrix, x,y,z,w): {calculated_quat}")
    
    deg_xyzw, deg_wxyz = compare_quat(calculated_quat, target_quat)
    
    match_found = False
    print("\n--- Validation Results ---")
    if deg_xyzw < 1.0:
        print(f"[MATCH] Matches as (x, y, z, w) convention. Diff: {deg_xyzw:.6f} degrees")
        match_found = True
    else:
         print(f"[NO MATCH] As (x, y, z, w). Diff: {deg_xyzw:.6f} degrees")

    if deg_wxyz < 1.0:
        print(f"[MATCH] Matches as (w, x, y, z) convention. Diff: {deg_wxyz:.6f} degrees")
        match_found = True
    else:
         print(f"[NO MATCH] As (w, x, y, z). Diff: {deg_wxyz:.6f} degrees")
         
    if not match_found:
        print(">> WARNING: The matrix rotation does not match the provided quaternion in either convention.")
        # Check for magnitude match (coordinate system issue)
        # Check against x,y,z,w
        if np.allclose(np.sort(np.abs(calculated_quat)), np.sort(np.abs(target_quat)), atol=0.01):
            print(">> NOTE: The vector components have similar magnitudes. This might be a coordinate system issue (sign flips or axis swaps).")
    
    return translation, r

# ==========================================
# 1. First Set (Official)
# ==========================================
tf_official = np.array([
    7.416679444534866883e-02,-9.902696855667120213e-01,1.177507386359286923e-01,-7.236400044878017468e-01,
    -1.274026398887237732e-01,1.076995435286611930e-01,9.859864987275952508e-01,-6.886495877727516479e-01,
    -9.890742408692511090e-01,-8.812921292808308105e-02,-1.181752422362273985e-01,6.366771698474239516e-01,
    0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,1.000000000000000000e+00
]).reshape(4,4)

target_quat_official = np.array([ 0.51567701, -0.52073085,  0.53658829,  0.41831759])

trans_official, rot_official = analyze_pose("Official Set", tf_official, target_quat_official)

print("\n")

# ==========================================
# 2. Second Set (User's)
# ==========================================
tf_user = np.array([
    0.000,  -0.662,  0.749,  0.113,
   -1.000,   0.000,  0.000,  0.011,
    0.000,  -0.749, -0.662,  1.553,
    0.000,   0.000,  0.000,  1.000
]).reshape(4, 4)

# Provided quaternion
target_quat_user = np.array([-0.645, -0.645, -0.291, 0.291])

trans_user, rot_user = analyze_pose("User's Set", tf_user, target_quat_user)

print("\n")
print(f"==================================================")
print(f" COMPARISON: Official vs User")
print(f"==================================================")

# Translation Difference
diff_trans_vec = trans_official - trans_user
dist = np.linalg.norm(diff_trans_vec)
print(f"Official Translation: {trans_official}")
print(f"User Translation:     {trans_user}")
print(f"Relative Translation Vector: {diff_trans_vec}")
print(f"Euclidean Distance: {dist:.6f} meters")

# Rotation Difference
# Relative rotation R_rel = R_official * inv(R_user)
# This represents the rotation needed to go from User frame to Official frame
rot_diff = rot_official * rot_user.inv()
angle_diff = rot_diff.magnitude() # in radians
angle_diff_deg = np.degrees(angle_diff)

print(f"\nOfficial Rotation (x,y,z,w): {rot_official.as_quat()}")
print(f"User Rotation (x,y,z,w):     {rot_user.as_quat()}")
print(f"Rotation Difference (Angular Distance): {angle_diff_deg:.6f} degrees")
