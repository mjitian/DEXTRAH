import argparse
import urdfpy
from pxr import Usd, UsdGeom, Sdf
import os
import tempfile
import trimesh  # 用于读取STL/OBJ的顶点和面数据
import numpy as np


def replace_ros_package_path(urdf_path, package_name, package_local_path):
    """
    替换URDF中的ROS package://路径为本地实际路径
    :param urdf_path: 原URDF文件路径
    :param package_name: ROS包名（如tiangong2pro）
    :param package_local_path: ROS包的本地路径（如/home/dodo/下载/tiangong2pro/）
    :return: 临时URDF文件的路径（替换后的内容）
    """
    # 读取原URDF内容
    with open(urdf_path, 'r', encoding='utf-8') as f:
        urdf_content = f.read()

    # 替换package://{package_name}/为本地路径
    ros_package_prefix = f"package://{package_name}/"
    # 确保本地路径以/结尾
    package_local_path = os.path.join(package_local_path, '')
    urdf_content = urdf_content.replace(ros_package_prefix, package_local_path)

    # 创建临时文件保存替换后的内容（避免修改原文件）
    temp_urdf = tempfile.NamedTemporaryFile(mode='w', suffix='.urdf', delete=False, encoding='utf-8')
    temp_urdf.write(urdf_content)
    temp_urdf.close()

    return temp_urdf.name


def urdf_to_usd(urdf_path, usd_path):
    # ========== 关键配置：替换为你的实际路径 ==========
    package_name = "tiangong2pro"  # URDF中的包名
    package_local_path = "/home/dodo/下载/tiangong2pro/"  # 包的本地根路径（包含meshes文件夹）

    # 生成替换后的临时URDF文件
    temp_urdf_path = replace_ros_package_path(urdf_path, package_name, package_local_path)
    print(f"使用替换路径后的临时URDF文件：{temp_urdf_path}")

    # 加载替换后的URDF
    print(f"加载URDF文件：{urdf_path}（已替换package路径）")
    urdf = urdfpy.URDF.load(temp_urdf_path)

    # 创建USD舞台
    stage = Usd.Stage.CreateNew(usd_path)
    root_prim = stage.DefinePrim("/robot", "Xform")
    stage.SetDefaultPrim(root_prim)

    # 遍历URDF的link，生成USD节点
    for link in urdf.links:
        # 创建link的Xform节点
        link_prim_path = f"/robot/{link.name}"
        link_prim = stage.DefinePrim(link_prim_path, "Xform")

        # 处理link中的visual
        for idx, visual in enumerate(link.visuals):
            # 为visual创建Mesh节点（如果有多个visual，用索引区分名称）
            visual_name = visual.name if visual.name else f"visual_{idx}"
            visual_prim_path = f"{link_prim_path}/{visual_name}"
            visual_mesh = UsdGeom.Mesh.Define(stage, visual_prim_path)

            # 处理mesh几何数据：读取STL/OBJ并设置顶点、面
            if visual.geometry.mesh is not None:
                mesh_file = visual.geometry.mesh.filename
                print(f"读取mesh文件：{mesh_file}")
                try:
                    # 用trimesh加载mesh文件（支持STL、OBJ等）
                    mesh = trimesh.load(mesh_file)

                    # 提取顶点和面数据（转换为USD要求的格式）
                    # 顶点：numpy数组，形状为(n, 3)，转换为USD的float3数组
                    points = mesh.vertices.astype(np.float32)
                    # 面：trimesh的faces是(n, 3)的数组，USD需要先指定每个面的顶点数（都是3），再指定顶点索引
                    face_vertex_counts = [3] * len(mesh.faces)  # 每个三角面有3个顶点
                    face_vertex_indices = mesh.faces.flatten().tolist()  # 展平为一维列表

                    # 设置USD Mesh的几何属性
                    visual_mesh.CreatePointsAttr(points)
                    visual_mesh.CreateFaceVertexCountsAttr(face_vertex_counts)
                    visual_mesh.CreateFaceVertexIndicesAttr(face_vertex_indices)

                    # 可选：设置几何的变换（visual的origin）
                    if visual.origin is not None:
                        # 获取visual的位姿矩阵，转换为USD的变换
                        transform = visual.origin
                        UsdGeom.Xformable(visual_mesh).AddTransformOp().Set(transform)

                except Exception as e:
                    print(f"警告：读取mesh文件{mesh_file}失败，错误：{e}")

    # 保存USD文件
    stage.Save()
    print(f"\n✅ USD文件已成功生成：{usd_path}")

    # 删除临时文件
    os.unlink(temp_urdf_path)
    print(f"🗑️ 临时文件已删除：{temp_urdf_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="URDF to USD converter (支持ROS package路径+STL/OBJ加载)")
    parser.add_argument("--input", required=True, help="URDF文件路径")
    parser.add_argument("--output", required=True, help="USD文件路径")
    args = parser.parse_args()
    urdf_to_usd(args.input, args.output)
