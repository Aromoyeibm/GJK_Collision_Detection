def replace_mesh_paths(input_urdf: str,
                       package: str,
                       package_path: str,
                       output_urdf: str):
    with open(input_urdf, "r") as f:
        text = f.read()
    text = text.replace(f"package://{package}/", f"file://{package_path}/")
    with open(output_urdf, "w") as f:
        f.write(text)
    print(f"Saved fixed URDF to {output_urdf}")
 