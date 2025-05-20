import sys

def clean_obj(input_path, output_path):
    with open(input_path, 'r') as f:
        lines = f.readlines()

    vertices = []
    faces = []
    other_lines = []

    # Parse vertices and faces
    for line in lines:
        if line.startswith('v '):
            vertices.append(line)
        elif line.startswith('f '):
            faces.append(line)
        else:
            other_lines.append(line)

    # Collect used vertex indices (1-based indexing in .obj)
    used_indices = set()
    for face in faces:
        parts = face.strip().split()[1:]
        for part in parts:
            v_idx = part.split('/')[0]
            used_indices.add(int(v_idx))

    # Map old vertex indices to new ones
    index_map = {}
    new_vertices = []
    for new_idx, old_idx in enumerate(sorted(used_indices), start=1):
        index_map[old_idx] = new_idx
        new_vertices.append(vertices[old_idx - 1])  # -1 because .obj is 1-based

    # Rebuild face definitions
    new_faces = []
    for face in faces:
        parts = face.strip().split()
        new_face = ['f']
        for part in parts[1:]:
            sub_parts = part.split('/')
            v = int(sub_parts[0])
            new_v = str(index_map[v])
            if len(sub_parts) == 1:
                new_face.append(new_v)
            elif len(sub_parts) == 2:
                new_face.append(f'{new_v}/{sub_parts[1]}')
            elif len(sub_parts) == 3:
                if sub_parts[1] == '':
                    new_face.append(f'{new_v}//{sub_parts[2]}')
                else:
                    new_face.append(f'{new_v}/{sub_parts[1]}/{sub_parts[2]}')
        new_faces.append(' '.join(new_face) + '\n')

    # Write cleaned file
    with open(output_path, 'w') as f:
        for line in other_lines:
            f.write(line)
        for v in new_vertices:
            f.write(v)
        for face in new_faces:
            f.write(face)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python clean_obj_unused_vertices.py input.obj output.obj")
        sys.exit(1)
    clean_obj(sys.argv[1], sys.argv[2])
