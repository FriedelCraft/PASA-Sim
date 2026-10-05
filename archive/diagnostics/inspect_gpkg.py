import trimesh

scene = trimesh.load("city.glb")

print(type(scene))

if isinstance(scene, trimesh.Scene):
    print("Geometry count:", len(scene.geometry))

    mesh = scene.dump(concatenate=True)
else:
    mesh = scene

print()
print("Vertices:", len(mesh.vertices))
print("Faces:", len(mesh.faces))
print("Watertight:", mesh.is_watertight)
print("Volume:", mesh.is_volume)
print("Euler:", mesh.euler_number)

edges = mesh.edges_unique
print("Unique edges:", len(edges))

