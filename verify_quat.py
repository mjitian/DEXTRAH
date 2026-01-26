import numpy as np
from scipy.spatial.transform import Rotation as R

def main():
    np.set_printoptions(precision=3, suppress=True)

    # 1. Create the rotation matrix from the array
    # Row-major interpretation
    tf = np.array([
        0.000,  -0.662,  0.749,  0.113,
       -1.000,   0.000,  0.000,  0.011,
        0.000,  -0.749, -0.662,  1.553,
        0.000,   0.000,  0.000,  1.000
    ]).reshape(4, 4)

    # Extract 3x3 rotation
    rot_matrix = tf[:3, :3]
    
    print("Input Rotation Matrix:")
    print(rot_matrix)
    print("=" * 60)

    # Target provided by user
    target_quat = np.array([-0.645, -0.645, -0.291, 0.291])

    # Helper to check a specific matrix
    def check(m, label):
        print(f"--- Checking {label} ---")
        r = R.from_matrix(m)
        
        # Scipy gives [x, y, z, w]
        q_xyzw = r.as_quat()
        # Construct [w, x, y, z] (Standard ordering for some libs like Isaac Sim/Gym)
        q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
        
        candidates = [
            ("Format [x, y, z, w]", q_xyzw),
            ("Format [w, x, y, z]", q_wxyz)
        ]
        
        for fmt_name, q_calc in candidates:
            # Check difference (considering sign ambiguity: q == -q)
            diff_pos = q_calc - target_quat
            diff_neg = q_calc + target_quat 
            
            dist_pos = np.linalg.norm(diff_pos)
            dist_neg = np.linalg.norm(diff_neg)
            
            # Select the closer one (sign wise) for display and comparison
            if dist_pos < dist_neg:
                best_q = q_calc
                dist = dist_pos
                sign_str = "(direct)"
            else:
                best_q = -q_calc
                dist = dist_neg
                sign_str = "(flipped sign)"
                
            match_str = "MATCH" if dist < 0.05 else "NO MATCH"
            
            print(f"{fmt_name}:")
            print(f"  Calculated {sign_str}: {best_q}")
            print(f"  Target:              {target_quat}")
            print(f"  Distance: {dist:.4f} -> {match_str}")
            if match_str == "NO MATCH":
                # Print element-wise diff to help debug axis swaps
                print(f"  Diff:   {np.abs(best_q - target_quat)}")
            print("-" * 40)

    check(rot_matrix, "Original Matrix (Row-Major)")
    
    # Also check Transpose in case of column-major mismatch
    check(rot_matrix.T, "Transposed Matrix (Column-Major)")

if __name__ == "__main__":
    main()
