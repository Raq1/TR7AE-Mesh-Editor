bl_info = {
    "name": "Tomb Raider LAU Mesh Editor",
    "author": "Raq",
    "collaborators": "Che, TheIndra, arcusmaximus (arc), DKDave, Joschka, Henry",
    "version": (1, 2, 5),
    "blender": (4, 2, 3),
    "location": "View3D > Sidebar > TRLAU Tools",
    "description": "Import and export Tomb Raider Legend/Anniversary/Underworld mesh files.",
    "category": "Import-Export",
}

import bpy
import struct
from math import radians
from mathutils import Vector, Quaternion
import random
import io
from pathlib import Path
from dataclasses import dataclass
from typing import List
from bpy.types import Operator, Panel
from bpy.props import StringProperty
from mathutils import Matrix
import tempfile
from bpy.app.handlers import persistent
from bpy.types import Operator, OperatorFileListElement, AddonPreferences
from bpy.props import IntProperty, FloatVectorProperty, CollectionProperty, PointerProperty
from bpy.props import StringProperty, CollectionProperty
import os
import subprocess
import bmesh
from bpy.props import EnumProperty
import bpy
from io import BytesIO
import mathutils
from mathutils import Euler
import numpy as np
import math

cloth_sim_state = {
    "positions": None,
    "velocities": None,
    "last_frame": -1
}

def is_power_of_two(n):
    return n > 0 and (n & (n - 1)) == 0

def convert_dds_to_pcd(dds_path, output_pcd_path, texture_id=None):
    dds_path = Path(dds_path)
    with open(dds_path, "rb") as f:
        header = f.read(128)
        if header[:4] != b'DDS ':
            raise ValueError("Invalid DDS file")

        height = struct.unpack_from("<I", header, 12)[0]
        width = struct.unpack_from("<I", header, 16)[0]
        mipmaps = struct.unpack_from("<I", header, 28)[0]
        if not (is_power_of_two(width) and is_power_of_two(height)):
            print(f"[PCD] Texture {dds_path.name} is NPOT ({width}x{height}) — disabling mipmaps.")
            mipmaps = 1
        fourcc = header[84:88]
        dxt_data = f.read()

    format_bytes = struct.unpack("<I", fourcc)[0]

    if texture_id is None:
        try:
            texture_name = dds_path.stem
            texture_id_hex = texture_name.split('_')[1]
            texture_id = int(texture_id_hex, 16)
        except Exception:
            raise ValueError("Failed to parse texture ID from filename. Use format INDEX_ID.dds")

    total_data_size = 24 + len(dxt_data)

    magic = 0x39444350  # 'PCD9'
    pcd_header = struct.pack(
        "<I i I I H H B B H",
        magic,
        format_bytes,
        len(dxt_data),
        0,
        width,
        height,
        0,
        mipmaps - 1,
        3
    )

    with open(output_pcd_path, "wb") as out:
        out.write(struct.pack("<4sI", b'SECT', total_data_size))
        out.write(struct.pack("<B", 5))
        out.write(b"\x00" * 7)
        out.write(struct.pack("<I", texture_id))
        out.write(struct.pack("<I", 0xFFFFFFFF))

        out.write(pcd_header)
        out.write(dxt_data)

def get_model_scale_from_bbox(armature):
    sorted_verts, _ = collect_and_sort_mesh_vertices(armature)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    bbox = BoundingBox.empty()

    for mesh, orig_idx, bone_ids in sorted_verts:
        eval_obj = mesh.evaluated_get(depsgraph)
        eval_mesh = eval_obj.to_mesh()
        v = eval_mesh.vertices[orig_idx]
        world_co = mesh.matrix_world @ v.co

        bone_index = bone_ids[0] if bone_ids else 0
        bone_name = f"Bone_{bone_index}"

        if bone_name in armature.data.bones:
            bone = armature.data.bones[bone_name]
            bone_world_matrix = armature.matrix_world @ bone.matrix_local
            bone_local_co = bone_world_matrix.inverted() @ world_co
        else:
            bone_local_co = world_co

        bbox.expand(bone_local_co)
        eval_obj.to_mesh_clear()

    size = bbox.max - bbox.min
    largest_axis = max(size.x, size.y, size.z)
    scale = (largest_axis / 32767,) * 3
    return scale

def find_closest_group_id(color):
    """
    Given an RGBA color, find the closest DryingGroup ID (0-31).
    """
    r, g, b = color[:3]

    closest_gid = None
    closest_distance = float("inf")

    for gid in range(32):
        random.seed(gid)
        gr, gg, gb = random.random(), random.random(), random.random()

        distance = (r - gr)**2 + (g - gg)**2 + (b - gb)**2

        if distance < closest_distance:
            closest_distance = distance
            closest_gid = gid

    return closest_gid

bpy.types.Object.tr7ae_show_mirror_data = bpy.props.BoolProperty(
    name="Show Bone Mirror Data",
    default=False
)

def collect_model_targets():
    targets = []
    for obj in bpy.context.scene.objects:
        if obj.get("tr7ae_type") == "ModelTarget":
            info = obj.tr7ae_modeltarget
            targets.append((
                obj.get("tr7ae_segment", 0),
                obj.get("tr7ae_flags", 0),
                info.px, info.py, info.pz,
                info.rx, info.ry, info.rz,
                info.unique_id
            ))
    return targets

def collect_and_sort_mesh_vertices(armature):
    from collections import defaultdict

    meshes = [obj for obj in armature.children
            if obj.type == 'MESH'
            and not obj.get("tr7ae_is_mface")]
    if not meshes:
        return [], {}

    bone_indices = {b.name: i for i, b in enumerate(armature.data.bones)}

    single_by_bone = {i: [] for i in bone_indices.values()}
    two_weight = []

    depsgraph = bpy.context.evaluated_depsgraph_get()
    for mesh in meshes:
        eval_obj = mesh.evaluated_get(depsgraph)
        eval_mesh = eval_obj.to_mesh()

        vg_lookup = {g.index: g.name for g in mesh.vertex_groups}

        for v in eval_mesh.vertices:
            bone_weight_pairs = []
            for g in v.groups:
                if g.weight > 0.001:
                    name = vg_lookup.get(g.group)
                    if name in bone_indices:
                        bone_index = bone_indices[name]
                        bone_weight_pairs.append((bone_index, g.weight))

            bone_weight_pairs.sort(key=lambda x: x[1], reverse=True)
            bone_ids = [pair[0] for pair in bone_weight_pairs]

            if len(bone_ids) > 2:
                raise ValueError(f"Mesh '{mesh.name}' has more than 2 weights per vertex.\n\nSelect the Armature, then select Utilities > Limit Weights and Normalize All.")

            if len(bone_ids) == 1:
                single_by_bone[bone_ids[0]].append((mesh, v.index, bone_ids))
            elif len(bone_ids) == 2:
                two_weight.append((mesh, v.index, bone_ids, bone_weight_pairs))

        eval_obj.to_mesh_clear()

    sorted_verts = []
    bone_ranges = {}
    cursor = 0

    for bone_id in range(len(armature.data.bones)):
        verts = single_by_bone.get(bone_id, [])
        count = len(verts)
        if count:
            sorted_verts.extend(verts)
            bone_ranges[bone_id] = (cursor, cursor + count - 1)
            cursor += count

    def sort_key(item):
        mesh, v_idx, bone_ids, bone_weights = item
        b0, b1 = bone_ids
        w0 = bone_weights[0][1]
        q_weight = round(w0 * 15) / 15
        return (b0, b1, q_weight, mesh.name, v_idx)

    two_weight_sorted = sorted(two_weight, key=sort_key)

    two_weight_sorted = [(mesh, v_idx, bone_ids) for mesh, v_idx, bone_ids, _ in two_weight_sorted]


    sorted_verts.extend(two_weight_sorted)

    return sorted_verts, bone_ranges


def collect_virtsegment_entries(sorted_verts, armature, quantize=False, levels=31):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    entries = []

    single_count = sum(1 for _, _, ids in sorted_verts if len(ids) == 1)

    run_key = None
    run_start = None

    for new_idx, (mesh, orig_idx, bone_ids) in enumerate(sorted_verts):
        if new_idx < single_count:
            continue

        b0, b1 = bone_ids
        eval_obj = mesh.evaluated_get(depsgraph)
        eval_mesh = eval_obj.to_mesh()

        vg = mesh.vertex_groups[f"Bone_{b1}"]
        w = 0.0
        for g in eval_mesh.vertices[orig_idx].groups:
            if g.group == vg.index:
                w = g.weight
                break
        eval_obj.to_mesh_clear()

        if quantize:
            w = round(w * levels) / levels
        else:
            w = round(w, 4)

        key = (b0, b1, w)
        if key != run_key:
            if run_key is not None:
                entries.append((run_start, new_idx - 1, *run_key))
            run_key = key
            run_start = new_idx

    if run_key:
        entries.append((run_start, len(sorted_verts) - 1, *run_key))

    return entries


from dataclasses import dataclass, field

@dataclass

class BoundingBox:
    min: Vector
    max: Vector

    @staticmethod
    def empty():
        inf = float('inf')
        return BoundingBox(Vector((inf, inf, inf)), Vector((-inf, -inf, -inf)))

    def expand(self, point: Vector):
        self.min.x = min(self.min.x, point.x)
        self.min.y = min(self.min.y, point.y)
        self.min.z = min(self.min.z, point.z)
        self.max.x = max(self.max.x, point.x)
        self.max.y = max(self.max.y, point.y)
        self.max.z = max(self.max.z, point.z)

@dataclass

class HInfoEntry:
    bone_index: int
    spheres: list = field(default_factory=list)
    hboxes: list = field(default_factory=list)
    hmarkers: list = field(default_factory=list)
    hcapsules: list = field(default_factory=list)
    offset: int = 0

def collect_structured_hinfo_data(armature):
    entries = []
    for i, bone in enumerate(armature.data.bones):
        pbone = armature.pose.bones.get(bone.name)
        if not pbone:
            continue

        entry = HInfoEntry(bone_index=i)

        for s in getattr(pbone, "tr7ae_hspheres", []):
            if s.radius <= 0:
                continue
            entry.spheres.append((
                int(s.flags),
                int(s.id),
                int(s.rank),
                int(round(s.radius)),
                int(round(s.x)), int(round(s.y)), int(round(s.z)),
                int(round(s.radius_sq)),
                int(round(s.mass)),
                int(round(s.buoyancy_factor)),
                int(round(s.explosion_factor)),
                int(round(s.material_type)),
                int(round(s.pad)),
                int(round(s.damage)),
            ))

        for h in getattr(pbone, "tr7ae_hboxes", []):
            entry.hboxes.append((
                h.widthx,
                h.widthy,
                h.widthz,
                h.widthw,
                h.positionboxx,
                h.positionboxy,
                h.positionboxz,
                h.positionboxw,
                h.quatx, h.quaty, h.quatz, h.quatw,
                int(h.flags),
                int(h.id),
                int(h.rank),
                int(round(h.mass)),
                int(round(h.buoyancy_factor)),
                int(round(h.explosion_factor)),
                int(h.material_type),
                int(h.pad),
                int(h.damage),
                int(h.pad1),
            ))

        for c in getattr(pbone, "tr7ae_hcapsules", []):
            if c.flags == 0:
                continue
            entry.hcapsules.append((
                c.posx,
                c.posy,
                c.posz,
                c.posw,
                c.quatx, c.quaty, c.quatz, c.quatw,
                int(c.flags),
                int(c.id),
                int(c.rank),
                int(round(c.radius)),
                int(round(c.length)),
                int(round(c.mass)),
                int(round(c.buoyancy_factor)),
                int(round(c.explosion_factor)),
                int(c.material_type),
                int(c.pad),
                int(c.damage),
            ))

        marker_list = getattr(pbone, "tr7ae_hmarkers", [])
        for m in marker_list:
            marker_obj = bpy.data.objects.get(f"HMarker_{m.bone}_{m.index}")
            if marker_obj:
                euler = marker_obj.rotation_euler
                rx, ry, rz = euler.x, euler.y, euler.z
            else:
                rx, ry, rz = m.rx, m.ry, m.rz

            entry.hmarkers.append((
                m.bone, m.index,
                m.px, m.py, m.pz,
                rx, ry, rz
            ))

        if entry.spheres or entry.hboxes or entry.hcapsules or entry.hmarkers:
            entries.append(entry)

    return entries

def export_texture_as_pcd(image, texture_id, export_dir, context):
    if not image or not image.filepath:
        return

    input_path = Path(bpy.path.abspath(image.filepath))
    if not input_path.exists():
        print(f"[Textures] Skipping missing image: {input_path}")
        return

    section_list_path = Path(export_dir) / "sectionList.txt"
    section_lines = []
    if section_list_path.exists():
        section_lines = section_list_path.read_text(encoding='utf-8').splitlines()

    used_indices = set()
    for line in section_lines:
        if "_" in line:
            try:
                index = int(line.split("_")[0])
                used_indices.add(index)
            except ValueError:
                pass
    next_index = max(used_indices, default=-1) + 1

    if texture_id is None:
        raise ValueError("Missing texture ID")

    hex_id = f"{texture_id:x}"

    existing_filename = None
    for line in section_lines:
        if line.strip().endswith(f"_{hex_id}.pcd"):
            existing_filename = line.strip()
            break

    if existing_filename:
        new_filename = existing_filename
    else:
        new_filename = f"{next_index}_{hex_id}.pcd"
        section_lines.append(new_filename)
        section_list_path.write_text("\n".join(section_lines) + "\n", encoding='utf-8')
        print(f"[Textures] Added new section: {new_filename}")

    new_path = Path(export_dir) / new_filename

    if input_path.suffix.lower() == ".dds":
        temp_dds = input_path
    else:
        temp_dds = Path(tempfile.gettempdir()) / (input_path.stem + ".dds")
        prefs = context.preferences.addons[__name__].preferences
        texconv_path = getattr(prefs, 'texconv_path', '') or "texconv.exe"
        format = "DXT1" if is_image_fully_opaque_blender(input_path) else "DXT5"

        if not os.path.exists(texconv_path):
            raise RuntimeError(f"Cannot convert non-DDS textures without a valid path to texconv.exe.\n\nSet a valid texconv.exe path in the TRLAU Mesh Editor Addon Preferences.")

        if not input_path.exists():
            raise RuntimeError(f"Texture file does not exist: {input_path}")

        try:
            result = subprocess.run([
                texconv_path,
                "-nologo",
                "-y",
                "-ft", "dds",
                "-f", format,
                "-o", str(temp_dds.parent),
                str(input_path)
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        except FileNotFoundError:
            raise RuntimeError(f"Failed to launch texconv.exe: File not found.\nMake sure texconv.exe is accessible at: {texconv_path}")
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.decode().strip()
            stdout = e.stdout.decode().strip()
            raise RuntimeError(f"texconv failed:\n{stderr or '(no stderr output)'}\n{stdout or ''}")

        if not temp_dds.exists():
            raise RuntimeError(f"Expected output DDS was not created: {temp_dds}")


    convert_dds_to_pcd(temp_dds, new_path, texture_id)
    print(f"[Textures] Exported {input_path.name} → {new_path.name}")



def export_used_textures(armature, model_export_path, context):
    exported_ids = set()
    base_dir = Path(model_export_path).parent

    for obj in bpy.data.objects:
        if obj.type != 'MESH' or obj.parent != armature:
            continue

        for mat_slot in obj.material_slots:
            mat = mat_slot.material
            if not mat or not mat.use_nodes:
                image_nodes = [
                    n for n in mat.node_tree.nodes
                    if n.type == 'TEX_IMAGE' and n.image is not None
                ]

                if len(image_nodes) > 1:
                    print(f"[WARNING] Material '{mat.name}' has multiple textures assigned — only one is supported.")
                continue

            tex_id = mat.get("tr7ae_texture_id")
            if tex_id is None or tex_id in exported_ids:
                continue

            for node in mat.node_tree.nodes:
                if node.type == 'TEX_IMAGE' and node.image:
                    export_texture_as_pcd(node.image, tex_id, base_dir, context)
                    exported_ids.add(tex_id)
                    break

def align_stream(mb, alignment):
    padding = (alignment - (mb.tell() % alignment)) % alignment
    if padding:
        mb.write(b"\x00" * padding)

class TR7AE_OT_ExportOldGenModel(Operator):
    bl_idname = "tr7ae.export_oldgen_model"
    bl_label = "Export TR7AE Model"
    bl_description = "Export TR7AE Model"
    bl_options = {'REGISTER'}

    filepath: StringProperty(subtype="FILE_PATH")

    export_hinfo: bpy.props.BoolProperty(
        name="Export HInfo",
        description="Include HSphere/HBox/HMarker/HCapsule data",
        default=True
    )

    export_textures: bpy.props.BoolProperty(
        name="Export Textures",
        description="Automatically export and replace texture .pcd files used by materials",
        default=True
    )

    filter_glob: bpy.props.StringProperty(
        default="*.tr7aemesh",
        options={'HIDDEN'}
    )

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "export_hinfo")
        layout.prop(self, "export_textures")

    def execute(self, context):
        offset_vertex_color = None
        gnc_data = bytearray()
        armature = context.active_object
        if not armature or armature.type != 'ARMATURE':
            self.report({'ERROR'}, "Please select a valid Armature.")
            return {'CANCELLED'}

        try:
            with open(self.filepath, "wb") as f:
                mb = io.BytesIO()
                relocations = []

                def write_offset(mb, value, reloc_list):
                    offset = mb.tell()
                    reloc_list.append(offset)
                    mb.write(struct.pack("<I", value))

                sorted_verts, bone_vertex_ranges = collect_and_sort_mesh_vertices(armature)
                if len(sorted_verts) > 21845:
                    self.report({'WARNING'}, f"Model has {len(sorted_verts)} vertices — game limit is 21845!")
                mesh_objs = [
                    obj for obj in bpy.context.scene.objects
                    if obj.parent == armature and obj.type == 'MESH'
                ]
                vertex_index_map = {
                    (mesh.name, orig_idx): i
                    for i, (mesh, orig_idx, _) in enumerate(sorted_verts)
                }
                flat_shading_used = False
                for mesh_obj in mesh_objs:
                    if mesh_obj.material_slots:
                        for slot in mesh_obj.material_slots:
                            material = slot.material
                            if material and material.get("tr7ae_flat_shading", 0) == 1:
                                flat_shading_used = True
                                break
                    if flat_shading_used:
                        break

                virt_entries = collect_virtsegment_entries(sorted_verts, armature, quantize=False)

                if flat_shading_used:
                    if len(virt_entries) > 153:
                        print("[INFO] Flat Shading detected and model has too many weights — quantizing weights to reduce virtSegments.")
                        for levels in [31, 15, 7, 3, 1]:
                            quantized = collect_virtsegment_entries(sorted_verts, armature, quantize=True, levels=levels)
                            if len(quantized) <= 153:
                                virt_entries = quantized
                                break
                        else:
                            raise ValueError("Could not reduce virtSegments to 153 or fewer with quantization.")

                else:
                    virt_entries = collect_virtsegment_entries(sorted_verts, armature, quantize=False)


                virt_segment_lookup = {}

                base_index = len(armature.data.bones)

                for i, (start, end, b0, b1, w) in enumerate(virt_entries):
                    for v in range(start, end + 1):
                        virt_segment_lookup[v] = base_index + i

                num_virtsegments = len(virt_entries)

                version = 79823955
                bones = armature.data.bones
                num_bones = len(bones)
                bone_data_offset_pos = f.tell() + 12
                mb.write(struct.pack("<4I", version, num_bones, num_virtsegments, 0))

                depsgraph = bpy.context.evaluated_depsgraph_get()
                bbox = BoundingBox.empty()

                for mesh, orig_idx, bone_ids in sorted_verts:
                    eval_obj = mesh.evaluated_get(depsgraph)
                    eval_mesh = eval_obj.to_mesh()
                    v = eval_mesh.vertices[orig_idx]

                    world_co = mesh.matrix_world @ v.co

                    bone_index = bone_ids[0] if bone_ids else 0
                    bone_name = f"Bone_{bone_index}"

                    if bone_name in armature.data.bones:
                        bone = armature.data.bones[bone_name]
                        bone_world_matrix = armature.matrix_world @ bone.matrix_local
                        bone_local_co = bone_world_matrix.inverted() @ world_co
                    else:
                        bone_local_co = world_co

                    bbox.expand(bone_local_co)
                    eval_obj.to_mesh_clear()

                size = bbox.max - bbox.min
                largest_axis = max(size.x, size.y, size.z)
                scale_value = largest_axis / 32767
                scale = (scale_value, scale_value, scale_value, 1.0)

                mb.seek(bone_data_offset_pos + 4)
                mb.write(struct.pack("<4f", *scale))


                max_rad = armature.get("max_rad", 0.0)
                max_rad_sq = max_rad * max_rad
                cdcRenderDataID = armature.get("cdcRenderDataID")

                wrote_vertex_colors = False
                wrote_env_mapped = False
                wrote_eye_ref = False
                vertex_header_offset = mb.tell()
                mb.write(struct.pack("<iI", 0, 0))  # num_vertices, vertex_list
                mb.write(struct.pack("<iI", 0, 0))  # num_normals, normal_list
                num_faces_offset = mb.tell()
                mb.write(struct.pack("<i", 0))  # placeholder for num_faces
                mb.write(struct.pack("<I", 0))      # mface_list
                mb.write(struct.pack("<I", 0))      # OBSOLETE aniTextures
                mb.write(struct.pack("<2f", max_rad, max_rad_sq))  # max_rad, max_rad_sq
                mb.write(struct.pack("<I", 0))      # OBSOLETE startTextures
                mb.write(struct.pack("<I", 0))      # OBSOLETE endTextures
                mb.write(struct.pack("<I", 0))      # animatedListInfo
                mb.write(struct.pack("<I", 0))      # animatedInfo
                mb.write(struct.pack("<I", 0))      # scrollInfo
                face_list_offset_pos = mb.tell()
                mb.write(struct.pack("<I", 0))      # face_list
                write_offset(mb, 0, relocations)
                write_offset(mb, 0, relocations)
                vertex_color_offset_pos = mb.tell()
                mb.write(struct.pack("<I", 0))      # vertex_color_offset
                mb.write(struct.pack("<I", 0))      # spectralVertexColors
                mb.write(struct.pack("<I", 0))      # pnShadowFaces
                mb.write(struct.pack("<I", 0))      # pnShadowEdges
                bone_mirror_data_offset_pos = mb.tell()
                mb.write(struct.pack("<I", 0))      # boneMirrorData
                mb.write(struct.pack("<I", 0))      # drawGroupCenterList
                mb.write(struct.pack("<I", 0))      # numMarkups
                mb.write(struct.pack("<I", 0))      # markupList
                num_targets_offset = mb.tell()
                mb.write(struct.pack("<I", 0))      # numTargets
                target_list_offset = mb.tell()
                mb.write(struct.pack("<I", 0))      # targetList
                mb.write(struct.pack("<I", cdcRenderDataID))  # cdcRenderDataID
                mb.write(struct.pack("<I", 0))  # cdcRenderDataID
                mb.write(struct.pack("<I", 0))      # cdcRenderModelData
                # Step: Write bone mirror data (if any)
                bone_mirror_data = armature.get("bone_mirror_data", [])
                if bone_mirror_data:
                    bone_mirror_offset = mb.tell()
                    mb.seek(bone_mirror_data_offset_pos)
                    write_offset(mb, bone_mirror_offset, relocations)
                    mb.seek(bone_mirror_offset)

                    for entry in bone_mirror_data:
                        b1 = entry.get("bone1", 0)
                        b2 = entry.get("bone2", 0)
                        count = entry.get("count", 0)
                        mb.write(struct.pack("<3B", b1, b2, count))

                    mb.write(struct.pack("<H", 0))

                sorted_verts, bone_vertex_ranges = collect_and_sort_mesh_vertices(armature)
                align_stream(mb, 16)
                bone_offset = mb.tell()
                mb.seek(bone_data_offset_pos)
                write_offset(mb, bone_offset, relocations)
                mb.seek(bone_offset)
                bone_index_map = {b.name: i for i, b in enumerate(bones)}

                if self.export_hinfo:
                    hinfo_entries = collect_structured_hinfo_data(armature)

                    hentry_by_index = {e.bone_index: e for e in hinfo_entries}

                bone_positions = []
                for i, bone in enumerate(bones):
                    pbone = armature.pose.bones.get(bone.name)

                    if bone.parent:
                        pivot = bone.head_local - bone.parent.head_local
                    else:
                        pivot = bone.head_local
                    pivot_q = (pivot.x, pivot.y, pivot.z, 0.0)

                    min_q = pbone.get("min_q", (0.0, 0.0, 0.0, 0.0))
                    max_q = pbone.get("max_q", (0.0, 0.0, 0.0, 0.0))
                    if isinstance(min_q, list): min_q = tuple(min_q)
                    if isinstance(max_q, list): max_q = tuple(max_q)

                    flags = pbone.get("flags", 0)
                    parent_index = bone_index_map[bone.parent.name] if bone.parent else -1

                    first_vert, last_vert = bone_vertex_ranges.get(i, (0, -1))

                    mb.write(struct.pack("<4f", *min_q))
                    mb.write(struct.pack("<4f", *max_q))
                    bone_positions.append(pivot)
                    mb.write(struct.pack("<4f", *pivot_q))
                    mb.write(struct.pack("<i", flags))
                    mb.write(struct.pack("<h", first_vert))
                    mb.write(struct.pack("<h", last_vert))
                    mb.write(struct.pack("<i", parent_index))
                    mb.write(struct.pack("<I", 0))  # placeholder for HInfo offset


                align_stream(mb, 16)
                num_virtsegments = len(virt_entries)

                for first, last, b0, b1, w in virt_entries:
                    mb.write(struct.pack("<4f", 0.0, 0.0, 0.0, 0.0))
                    mb.write(struct.pack("<4f", 0.0, 0.0, 0.0, 0.0))

                    def compute_virtual_segment_pivot(b0, b1, bones, write_bone_index):
                        bone_a = bones[b0]
                        bone_b = bones[b1]

                        use_inverse = (write_bone_index == b1)

                        parent = bone_a.parent if write_bone_index == b0 else bone_b.parent
                        if parent:
                            parent_inv = parent.matrix_local.inverted()
                        else:
                            parent_inv = Matrix.Identity(4)

                        mat_a = bone_a.matrix_local @ parent_inv
                        mat_b = bone_b.matrix_local @ parent_inv

                        pivot_vec = (mat_b.translation - mat_a.translation)

                        if use_inverse:
                            pivot_vec = -pivot_vec

                        return (pivot_vec.x, pivot_vec.y, pivot_vec.z, 1.0)
                    
                    pivot = compute_virtual_segment_pivot(b0, b1, bones, write_bone_index=b1)
                    mb.write(struct.pack("<4f", *pivot))

                    mb.write(struct.pack("<i", 8))
                    mb.write(struct.pack("<4h", first, last, b0, b1))
                    mb.write(struct.pack("<f", w))

                if self.export_hinfo:
                    for hentry in hinfo_entries:
                        hentry.offset = mb.tell()

                        hsphere_offset_pos = mb.tell() + 4
                        hbox_offset_pos = hsphere_offset_pos + 8
                        hmarker_offset_pos = hbox_offset_pos + 8
                        hcapsule_offset_pos = hmarker_offset_pos + 8

                        mb.write(struct.pack("<iI", len(hentry.spheres), 0))
                        mb.write(struct.pack("<iI", len(hentry.hboxes), 0))
                        mb.write(struct.pack("<iI", len(hentry.hmarkers), 0))
                        mb.write(struct.pack("<iI", len(hentry.hcapsules), 0))

                        def write_list(items, fmt, offset_pos):
                            if not items:
                                return
                            align_stream(mb, 16)
                            list_offset = mb.tell()
                            mb.seek(offset_pos)
                            write_offset(mb, list_offset, relocations)
                            mb.seek(list_offset)
                            for item in items:
                                mb.write(struct.pack(fmt, *item))

                        write_list(hentry.spheres, "<HBBHhhhIHB3B h", hsphere_offset_pos)
                        write_list(hentry.hboxes, "<4f4f4f hBBH BBBB hI", hbox_offset_pos)
                        write_list(hentry.hmarkers, "<ii6f", hmarker_offset_pos)
                        write_list(hentry.hcapsules, "<4f4f h bb HHhbbbb h ", hcapsule_offset_pos)

                    for hentry in hinfo_entries:
                        bone_struct_start = bone_offset + hentry.bone_index * 64
                        hinfo_field_offset = bone_struct_start + 60
                        mb.seek(hinfo_field_offset)
                        write_offset(mb, hentry.offset, relocations)

                if self.export_textures:
                    export_used_textures(armature, self.filepath, context)

                mb.seek(0, 2)
                vertex_list_offset = mb.tell()
                num_vertices = len(sorted_verts)
                mb.seek(vertex_header_offset)
                mb.write(struct.pack("<i", num_vertices))
                write_offset(mb, vertex_list_offset, relocations)
                mb.seek(vertex_list_offset)

                def float_to_ushort(f):
                    return struct.unpack("<I", struct.pack("<f", f))[0] >> 16

                def clamp_short(v):
                    return max(min(int(v), 32767), -32768)

                mb.seek(vertex_list_offset)

                depsgraph = bpy.context.evaluated_depsgraph_get()

                for obj in bpy.context.view_layer.objects:
                    if obj.type != 'MESH' or obj.parent != armature:
                        continue

                    eval_obj = obj.evaluated_get(depsgraph)
                    eval_mesh = eval_obj.to_mesh()

                    # Check for any non-triangle faces
                    needs_triangulate = any(poly.loop_total != 3 for poly in eval_mesh.polygons)

                    eval_obj.to_mesh_clear()

                    if needs_triangulate:
                        print(f"[INFO] Triangulating '{obj.name}' (non-triangular faces detected).")

                        # Add and apply triangulate modifier
                        triangulate_mod = obj.modifiers.new(name="TRIANGULATE_TEMP", type='TRIANGULATE')
                        triangulate_mod.show_expanded = False

                        # Make sure object is active for applying modifier
                        bpy.context.view_layer.objects.active = obj
                        bpy.ops.object.modifier_apply(modifier=triangulate_mod.name)

                    # Add vertex colors if missing
                    mesh = obj.data
                    if not mesh.color_attributes:
                        print(f"[INFO] Adding Vertex Colors to '{obj.name}'.")
                        color_layer = mesh.color_attributes.new(name="Color", type='BYTE_COLOR', domain='CORNER')
                        for color in color_layer.data:
                            color.color = (1.0, 1.0, 1.0, 1.0)

                for v_idx, (mesh, orig_idx, bone_ids) in enumerate(sorted_verts):
                    eval_obj = mesh.evaluated_get(depsgraph)
                    eval_mesh = eval_obj.to_mesh()
                    v = eval_mesh.vertices[orig_idx]

                    world_co = mesh.matrix_world @ v.co

                    bone_index = bone_ids[0] if bone_ids else 0
                    bone_name = f"Bone_{bone_index}"

                    if bone_name in armature.data.bones:
                        bone = armature.data.bones[bone_name]
                        bone_world_matrix = armature.matrix_world @ bone.matrix_local
                        bone_local_co = bone_world_matrix.inverted() @ world_co
                    else:
                        bone_local_co = world_co

                    origin = Vector((0.0, 0.0, 0.0))
                    normalized = (bone_local_co - origin) / largest_axis

                    x = clamp_short(normalized.x * 32767)
                    y = clamp_short(normalized.y * 32767)
                    z = clamp_short(normalized.z * 32767)

                    layer_color = None
                    for vcol_layer in mesh.data.vertex_colors:
                        if vcol_layer.name != "MFace":
                            layer_color = vcol_layer
                            break

                    if layer_color:
                        color_data = layer_color.data

                        for loop in eval_mesh.loops:
                            if loop.vertex_index == orig_idx:
                                col = color_data[loop.index].color
                                r = int(col[0] * 128)
                                g = int(col[1] * 128)
                                b = int(col[2] * 128)
                                a = int(col[3] * 128) if len(col) == 4 else 255
                                gnc_data.extend([b, g, r, a])
                                wrote_vertex_colors = True
                                break


                    loop_normal = None
                    for loop in eval_mesh.loops:
                        if loop.vertex_index == orig_idx:
                            loop_normal = loop.normal
                            break

                    if loop_normal:
                        no = loop_normal.normalized()
                    else:
                        no = v.normal.normalized()  # Fallback just in case

                    nx = int(max(min(no.x * 127, 127), -128))
                    ny = int(max(min(no.y * 127, 127), -128))
                    nz = int(max(min(no.z * 127, 127), -128))

                    if v_idx in virt_segment_lookup:
                        segment = virt_segment_lookup[v_idx]
                    else:
                        segment = bone_ids[0] if bone_ids else 0
                    uv = (0.0, 0.0)
                    uv_layer = eval_mesh.uv_layers.active
                    if uv_layer:
                        for loop in eval_mesh.loops:
                            if loop.vertex_index == orig_idx:
                                uv_data = uv_layer.data[loop.index].uv
                                uv = (uv_data.x, uv_data.y)
                                break

                    uvx = float_to_ushort(uv[0])
                    uvy = float_to_ushort(1.0 - uv[1])

                    mb.write(struct.pack("<3h", x, y, z))       # position
                    mb.write(struct.pack("<3b", nx, ny, nz))    # normal
                    mb.write(struct.pack("<B", 0))              # pad
                    mb.write(struct.pack("<h", segment))        # bone index
                    mb.write(struct.pack("<2H", uvx, uvy))      # UV

                    eval_obj.to_mesh_clear()

                align_stream(mb, 16)
                mface_list_offset = mb.tell()

                wrote_mface_data = False

                for mesh_obj in mesh_objs:
                    if mesh_obj.get("tr7ae_is_mface"):
                        continue

                    depsgraph = bpy.context.evaluated_depsgraph_get()
                    eval_obj = mesh_obj.evaluated_get(depsgraph)
                    eval_mesh = eval_obj.to_mesh()

                    drying_layer = eval_mesh.vertex_colors.get("MFace")
                    if not drying_layer:
                        continue

                    for poly in eval_mesh.polygons:
                        if poly.loop_total != 3:
                            continue

                        indices = []
                        gids = []

                        for i in range(3):
                            loop = eval_mesh.loops[poly.loop_start + i]
                            vertex_index = loop.vertex_index
                            mapped_idx = vertex_index_map.get((mesh_obj.name, vertex_index))
                            if mapped_idx is None:
                                break
                            indices.append(mapped_idx)

                            color = drying_layer.data[poly.loop_start + i].color
                            gid = find_closest_group_id(color)
                            gids.append(gid)

                        if len(indices) != 3 or len(gids) != 3:
                            continue

                        if not wrote_mface_data:
                            align_stream(mb, 16)
                            mface_list_offset = mb.tell()
                            wrote_mface_data = True

                        v0, v1, v2 = indices
                        gid0, gid1, gid2 = gids

                        gid0 = max(gid0, 1)
                        gid1 = max(gid1, 1)
                        gid2 = max(gid2, 1)

                        same = (gid2 << 10) | (gid1 << 5) | gid0
                        mb.write(struct.pack("<4H", v0, v1, v2, same))


                if wrote_mface_data:
                    mb.seek(num_faces_offset + 4)
                    write_offset(mb, mface_list_offset, relocations)

                mb.seek(0, 2)

                eval_obj.to_mesh_clear()

                align_stream(mb, 16)
                face_list_offset = mb.tell()

                mb.seek(face_list_offset_pos)
                write_offset(mb, face_list_offset, relocations)
                mb.seek(face_list_offset)

                chunk_starts = []

                total_faces = 0

                envmapped_vertex_indices = set()
                eye_ref_vertex_indices = set()

                for mesh_obj in mesh_objs:
                    if mesh_obj.get("tr7ae_is_mface"):
                        continue

                    depsgraph = bpy.context.evaluated_depsgraph_get()
                    eval_obj = mesh_obj.evaluated_get(depsgraph)
                    eval_mesh = eval_obj.to_mesh()

                    material_to_indices = {}

                    for poly in eval_mesh.polygons:
                        if poly.loop_total != 3:
                            continue

                        mat_index = poly.material_index
                        if mat_index not in material_to_indices:
                            material_to_indices[mat_index] = []

                        tri = []
                        for i in range(3):
                            loop_idx = poly.loop_start + i
                            v_idx = eval_mesh.loops[loop_idx].vertex_index
                            mapped_idx = vertex_index_map.get((mesh_obj.name, v_idx))
                            if mapped_idx is not None:
                                tri.append(mapped_idx)

                        if len(tri) == 3:
                            material_to_indices[mat_index].extend(tri)

                            if mesh_obj.data.get("tr7ae_is_envmapped", False):
                                envmapped_vertex_indices.update(tri)
                                wrote_env_mapped = True
                            if mesh_obj.data.get("tr7ae_is_eyerefenvmapped", False):
                                eye_ref_vertex_indices.update(tri)
                                wrote_eye_ref = True

                    eval_obj.to_mesh_clear()

                    for mat_index, indices in material_to_indices.items():
                        if not indices:
                            continue

                        align_stream(mb, 16)
                        chunk_starts.append(mb.tell())

                        draw_group = int(mesh_obj.data.get("tr7ae_draw_group", 0))
                        material = None
                        if mat_index < len(mesh_obj.material_slots):
                            material = mesh_obj.material_slots[mat_index].material

                        # use default if missing
                        texture_id   = int(material.get("tr7ae_texture_id", 0)) if material else 0
                        blend_value  = int(material.get("tr7ae_blend_value", 0)) if material else 0
                        unknown_1    = int(material.get("tr7ae_unknown_1", 0)) if material else 0
                        unknown_2    = int(material.get("tr7ae_unknown_2", 0)) if material else 0
                        single_sided = int(material.use_backface_culling) if material else 1
                        texture_wrap = int(material.get("tr7ae_texture_wrap", 0)) if material else 0
                        unknown_3    = int(material.get("tr7ae_unknown_3", 0)) if material else 0
                        unknown_4    = int(material.get("tr7ae_unknown_4", 0)) if material else 0
                        flat_shading = int(material.get("tr7ae_flat_shading", 0)) if material else 0
                        sort_z       = int(material.get("tr7ae_sort_z", 0)) if material else 0
                        stencil_pass = int(material.get("tr7ae_stencil_pass", 0)) if material else 0
                        stencil_func = int(material.get("tr7ae_stencil_func", 0)) if material else 0
                        alpha_ref    = int(material.get("tr7ae_alpha_ref", 0)) if material else 0

                        tpageid = (
                            (texture_id   & 0x1FFF)      |
                            ((blend_value & 0xF)   << 13) |
                            ((unknown_1   & 0x7)   << 17) |
                            ((unknown_2   & 0x1)   << 20) |
                            ((single_sided & 0x1) << 21) |
                            ((texture_wrap & 0x3) << 22) |
                            ((unknown_3   & 0x1)   << 24) |
                            ((unknown_4   & 0x1)   << 25) |
                            ((flat_shading & 0x1) << 26) |
                            ((sort_z      & 0x1)   << 27) |
                            ((stencil_pass & 0x3) << 28) |
                            ((stencil_func & 0x1) << 30) |
                            ((alpha_ref   & 0x1)   << 31)
                        )

                        vertex_count = len(indices)
                        face_count = vertex_count // 3
                        total_faces += face_count

                        mb.write(struct.pack("<2h", vertex_count, draw_group))
                        mb.write(struct.pack("<I", tpageid))
                        mb.write(b"\x00" * 8)
                        mb.write(struct.pack("<I", 0))
                        mb.write(struct.pack(f"<{vertex_count}H", *indices))

                last_real_chunk_offset = chunk_starts[-1]
                terminator_offset = mb.tell()
                mb.write(b"\x00" * 0x14)
                mb.seek(last_real_chunk_offset + 0x10)
                write_offset(mb, terminator_offset, relocations)

                for i in range(len(chunk_starts) - 1):
                    this_chunk_offset = chunk_starts[i]
                    next_chunk_offset = chunk_starts[i + 1]

                    mb.seek(this_chunk_offset + 0x10)
                    write_offset(mb, next_chunk_offset, relocations)

                mb.seek(num_faces_offset)
                mb.write(struct.pack("<i", total_faces))
                targets = collect_model_targets()
                target_gnc_offset = len(gnc_data)
                if targets:
                    mb.seek(target_list_offset)
                    write_offset(mb, target_list_offset, relocations)
                    mb.seek(target_list_offset)
                    mb.write(struct.pack("<I", target_gnc_offset + 8))
                    mb.seek(0, 2)
                    mb.seek(num_targets_offset)
                    mb.write(struct.pack("<I", len(targets)))
                    mb.seek(0, 2)

                if targets:
                    for seg, flags, px, py, pz, rx, ry, rz, uid in targets:
                        gnc_data.extend(struct.pack("<HH", seg, flags))
                        gnc_data.extend(struct.pack("<3f", px, py, pz))
                        gnc_data.extend(struct.pack("<3f", rx, ry, rz))
                        gnc_data.extend(struct.pack("<I", uid))

                model_data = bytearray(mb.getvalue())

                if wrote_vertex_colors:
                    vertex_color_offset_in_gnc = 8
                    struct.pack_into("<I", model_data, 100, vertex_color_offset_in_gnc)
                    relocations.append(100)
                model_data_size = len(model_data)

                with open(self.filepath, "wb") as f:
                    f.write(b'SECT')
                    f.write(struct.pack("<I", model_data_size))
                    f.write(struct.pack("<I", 0))
                    packed_reloc = (len(relocations) << 8)
                    f.write(struct.pack("<I", packed_reloc))
                    f.write(struct.pack("<I", 0))
                    f.write(struct.pack("<I", 0xFFFFFFFF))
                    while mb.tell() < 100:
                        mb.write(b"\x00")
                    mb.write(struct.pack("<I", 0))


                    sections = armature.tr7ae_sections

                    for reloc in sorted(relocations):
                        section_index = sections.main_file_index or 0

                    for reloc in sorted(relocations):
                        if reloc in (92, 96, 100, 136) and sections.extra_file_index:
                            section_index = sections.extra_file_index
                        else:
                            section_index = sections.main_file_index or 0


                        f.write(struct.pack("<HHI", (section_index << 3), 0, reloc))

                    from pathlib import Path

                    sections = armature.tr7ae_sections
                    extra_index = sections.extra_file_index

                    if extra_index:
                        gnc_filename = f"{extra_index}_0.gnc"
                        gnc_path = Path(self.filepath).parent / gnc_filename

                        with open(gnc_path, "wb") as ef:
                            ef.write(b'SECT')
                            total_gnc_size = 8 + len(gnc_data)
                            if wrote_env_mapped:
                                total_gnc_size += 4 + len(envmapped_vertex_indices) * 2
                            if wrote_eye_ref:
                                total_gnc_size += 4 + len(eye_ref_vertex_indices) * 2

                            ef.write(struct.pack("<I", total_gnc_size))
                            ef.write(struct.pack("<I", 0))
                            ef.write(struct.pack("<I", 0))
                            ef.write(struct.pack("<I", 0))
                            ef.write(struct.pack("<I", 0xFFFFFFFF))
                            ef.write(struct.pack("<Q", 0))

                            vertex_color_offset_in_gnc = ef.tell()

                            ef.write(gnc_data)

                            if wrote_env_mapped:
                                env_offset_in_gnc = ef.tell()
                                env_list = sorted(envmapped_vertex_indices)
                                ef.write(struct.pack("<I", len(env_list)))
                                ef.write(struct.pack(f"<{len(env_list)}H", *env_list))

                            if wrote_eye_ref:
                                eye_ref_offset_in_gnc = ef.tell()
                                env_list = sorted(eye_ref_vertex_indices)
                                ef.write(struct.pack("<I", len(env_list)))
                                ef.write(struct.pack(f"<{len(env_list)}H", *env_list))

                            if wrote_env_mapped:
                                struct.pack_into("<I", model_data, 92, env_offset_in_gnc - 24)

                            if wrote_eye_ref:
                                struct.pack_into("<I", model_data, 96, eye_ref_offset_in_gnc - 24)



                    f.write(model_data)


            self.report({'INFO'}, f"Exported model to {self.filepath}")
            return {'FINISHED'}

        except Exception as e:
            self.report({'ERROR'}, f"Failed to export: {e}")
            return {'CANCELLED'}

    def invoke(self, context, event):
        self.filter_glob = "*.tr7aemesh"
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

def is_legend(filepath):
    folder = Path(filepath).parent
    return any(f.suffix == ".tr7mesh" for f in folder.iterdir())

def import_markups(filepath, markup_offset, num_markups, num_relocations, is_legend):
    import os
    from struct import unpack

    section_index = None

    with open(filepath, "rb") as f:
        f.seek(24)
        for _ in range(num_relocations):
            packed = int.from_bytes(f.read(2), "little")
            section_index_candidate = packed >> 3
            f.read(2)
            offset = int.from_bytes(f.read(4), "little")
            if offset == 128:
                section_index = section_index_candidate
                break

    if section_index is None:
        print("[Markup] No relocation at offset 128.")
        return []

    gnc_path = os.path.join(os.path.dirname(filepath), f"{section_index}_0.gnc")
    if not os.path.exists(gnc_path):
        print(f"[Markup] Missing GNC file: {gnc_path}")
        return []

    data_start = get_gnc_data_offset(gnc_path)
    markups = []

    with open(gnc_path, "rb") as f:
        f.seek(data_start + markup_offset)

        for _ in range(num_markups):
            if is_legend:
                OverrideMovementCamera = unpack("<i", f.read(4))[0]
                DTPCameraDataID = unpack("<i", f.read(4))[0]
                DTPMarkupDataID = unpack("<i", f.read(4))[0]
                flags = unpack("<I", f.read(4))[0]
                introID, markupID = unpack("<hh", f.read(4))
                px, py, pz = unpack("<3f", f.read(12))
                bbox = unpack("<6h", f.read(12))
                poly_offset = unpack("<I", f.read(4))[0]
                AnimatedSegment = None  # ← Not used in Legend
            else:
                OverrideMovementCamera = unpack("<i", f.read(4))[0]
                DTPCameraDataID = unpack("<i", f.read(4))[0]
                DTPMarkupDataID = unpack("<i", f.read(4))[0]
                AnimatedSegment = unpack("<i", f.read(4))[0]
                cameraAntic = unpack("<6i", f.read(24))
                flags = unpack("<I", f.read(4))[0]
                introID, markupID = unpack("<hh", f.read(4))
                px, py, pz = unpack("<3f", f.read(12))
                bbox = unpack("<6h", f.read(12))
                poly_offset = unpack("<I", f.read(4))[0]
            print(f"[DEBUG] Seeking to polyline offset: {poly_offset + data_start} (poly_offset: {poly_offset})")

            cur = f.tell()
            f.seek(poly_offset + data_start)

            numPoints = unpack("<I", f.read(4))[0]
            f.read(12)

            points = []
            for _ in range(numPoints):
                x, y, z, _ = unpack("<4f", f.read(16))
                points.append((x, y, z))

            f.seek(cur)

            markups.append({
                "position": (px, py, pz),
                "flags": flags,
                "introID": introID,
                "markupID": markupID,
                "points": points,
            })

    print(f"[Markup] Imported {len(markups)} markups ({'Legend' if is_legend else 'Anniversary'})")
    return markups


def create_markup_visuals(markups, armature_obj):
    import bpy
    import bmesh
    from mathutils import Vector

    if "Markups" in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects["Markups"], do_unlink=True)

    parent = bpy.data.objects.new("Markups", None)
    parent.empty_display_type = 'CUBE'
    parent.empty_display_size = 0.5
    bpy.context.collection.objects.link(parent)
    parent.parent = armature_obj
    parent.matrix_parent_inverse.identity()

    for i, markup in enumerate(markups):
        points = markup["points"]
        if len(points) < 2:
            continue

        name = f"Markup_{i}"
        mesh_data = bpy.data.meshes.new(name)
        mesh_obj = bpy.data.objects.new(name, mesh_data)
        bpy.context.collection.objects.link(mesh_obj)

        verts = [Vector(p) for p in points]
        edges = [(j, j + 1) for j in range(len(verts) - 1)]

        mesh_data.from_pydata(verts, edges, [])
        mesh_data.update()

        mesh_obj["tr7ae_type"] = "Markup"
        mesh_obj["tr7ae_markupID"] = markup["markupID"]
        mesh_obj["tr7ae_introID"] = markup["introID"]
        mesh_obj["tr7ae_flags"] = markup["flags"]

        mesh_obj.parent = parent
        mesh_obj.matrix_parent_inverse.identity()

        mesh_obj.display_type = 'WIRE'
        mesh_obj.show_in_front = True



def int_to_flag_set(flag_value):
    return {f"{1<<i:#x}" for i in range(32) if flag_value & (1 << i)}

def import_model_targets(filepath, target_offset, num_targets, num_relocations):
    import os

    section_index = None
    with open(filepath, "rb") as f:
        f.seek(24)
        for _ in range(num_relocations):
            packed = int.from_bytes(f.read(2), "little")
            section_index_candidate = packed >> 3
            f.read(2)
            offset = int.from_bytes(f.read(4), "little")

            if offset == 136:
                section_index = section_index_candidate
                break

    gnc_path = os.path.join(os.path.dirname(filepath), f"{section_index}_0.gnc")
    if not os.path.exists(gnc_path):
        print(f"[WARNING] GNC file not found: {gnc_path}")
        return []

    targets = []
    with open(gnc_path, "rb") as f:
        from struct import unpack
        data_start = get_gnc_data_offset(gnc_path)
        f.seek(data_start + target_offset)

        for _ in range(num_targets):
            segment, flags = unpack("<HH", f.read(4))
            px, py, pz = unpack("<3f", f.read(12))
            rx, ry, rz = unpack("<3f", f.read(12))
            unique_id = unpack("<I", f.read(4))[0]

            targets.append({
                "segment": segment,
                "flags": flags,
                "position": (px, py, pz),
                "rotation": (rx, ry, rz),
                "uniqueId": unique_id
            })

    print(f"[ModelTarget] Loaded {len(targets)} model targets.")
    return targets

def create_model_target_visuals(targets, armature_obj):
    import bpy
    from math import radians
    from mathutils import Vector, Euler, Matrix

    if "Targets" in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects["Targets"], do_unlink=True)

    target_parent = bpy.data.objects.new("Targets", None)
    target_parent.empty_display_type = 'SPHERE'
    target_parent.empty_display_size = 0.3
    bpy.context.collection.objects.link(target_parent)
    target_parent.parent = armature_obj
    target_parent.matrix_parent_inverse.identity()

    for i, target in enumerate(targets):
        segment = target["segment"]
        px, py, pz = target["position"]
        rx, ry, rz = target["rotation"]
        uid = target["uniqueId"]

        bone_name = f"Bone_{segment}"
        if bone_name not in armature_obj.pose.bones:
            print(f"[WARNING] Bone '{bone_name}' not found for Target {i}")
            continue

        bpy.ops.mesh.primitive_torus_add(
            major_radius=0.05,
            minor_radius=0.01,
            major_segments=16,
            minor_segments=8,
            location=(0, 0, 0)
        )
        torus = bpy.context.active_object
        torus.name = f"Target_{uid}"
        torus.display_type = 'WIRE'
        torus.show_in_front = True
        torus["tr7ae_type"] = "ModelTarget"
        torus["tr7ae_segment"] = segment
        torus["tr7ae_uid"] = uid
        torus["tr7ae_flags"] = target["flags"]

        torus.rotation_euler = Euler((radians(90), 0, 0), 'XYZ')
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)

        torus.scale = (1000.0, 1000.0, 1000.0)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

        bone_mat = armature_obj.matrix_world @ armature_obj.data.bones[bone_name].matrix_local

        local_mat = Matrix.Translation(Vector((px, py, pz))) @ Euler((rx, ry, rz), 'XYZ').to_matrix().to_4x4()

        world_mat = bone_mat @ local_mat

        torus.matrix_world = target_parent.matrix_world.inverted() @ world_mat

        torus.parent = target_parent
        torus.matrix_parent_inverse.identity()

        flag_set = {f"{1<<i:#x}" for i in range(32) if target["flags"] & (1 << i)}
        torus.tr7ae_modeltarget.flags = flag_set
        torus.tr7ae_modeltarget.unique_id = uid
        torus.tr7ae_modeltarget.px = px
        torus.tr7ae_modeltarget.py = py
        torus.tr7ae_modeltarget.pz = pz
        torus.tr7ae_modeltarget.rx = rx
        torus.tr7ae_modeltarget.ry = ry
        torus.tr7ae_modeltarget.rz = rz


        vg = torus.vertex_groups.new(name=bone_name)
        vg.add(range(len(torus.data.vertices)), 1.0, 'REPLACE')

        arm_mod = torus.modifiers.new(name="Armature", type='ARMATURE')
        arm_mod.object = armature_obj
    
class TR7AE_SectionPaths(bpy.types.PropertyGroup):
    main_file_index: bpy.props.IntProperty(
        name="Main Section Index",
        description="Section index for the main mesh",
        default=0
    )
    extra_file_index: bpy.props.IntProperty(
        name="Extra Data Index",
        description="Section index for extra data (vertex colors, eye reflection/environment mapped vertices,\ntargets and markups)",
        default=0
    )
    cloth_file_index: bpy.props.IntProperty(
        name="Cloth Section Index",
        description="Section index for cloth physics",
        default=0
    )

    show_file_sections: bpy.props.BoolProperty(
        name="Show File Sections",
        description="Expand to show section index settings",
        default=False
    )

class TR7AE_HMarkerInfo(bpy.types.PropertyGroup):
    bone: bpy.props.IntProperty(name="Bone")
    index: bpy.props.IntProperty(name="Index")
    px: bpy.props.FloatProperty(name="Pos X")
    py: bpy.props.FloatProperty(name="Pos Y")
    pz: bpy.props.FloatProperty(name="Pos Z")
    rx: bpy.props.FloatProperty(name="Rot X")
    ry: bpy.props.FloatProperty(name="Rot Y")
    rz: bpy.props.FloatProperty(name="Rot Z")

TargetFlagItems = [
    ('0x1', "Combat", "", 1),
    ('0x2', "Incidental", "", 2),
    ('0x4', "Grapple Tug", "", 4),
    ('0x8', "Grapple Swing", "", 8),
    ('0x10', "Interact", "", 10),
    ('0x20', "Mag Gun", "", 20),
    ('0x40', "Forced Incidental", "", 40),
    ('0x80', "Aim Assist", "", 80),
    ('0x100', "Grapple Wall Run", "", 100),
    ('0x200', "Object", "", 200),
]

blend_mode_items = [
    ('0', "Opaque", "kPCBlendModeOpaque"),
    ('1', "Alpha Test", "kPCBlendModeAlphaTest"),
    ('2', "Alpha Blend", "kPCBlendModeAlphaBlend"),
    ('3', "Additive", "kPCBlendModeAdditive"),
    ('4', "Subtract", "kPCBlendModeSubtract"),
    ('5', "Dest Alpha", "kPCBlendModeDestAlpha"),
    ('6', "Dest Add", "kPCBlendModeDestAdd"),
    ('7', "Modulate", "kPCBlendModeModulate"),
    ('8', "Blend 50/50", "kPCBlendModeBlend5050"),
    ('9', "Dest Alpha Src Only", "kPCBlendModeDestAlphaSrcOnly"),
    ('10', "Color Modulate", "kPCBlendModeColorModulate"),
    ('13', "Multipass Alpha", "kPCBlendModeMultipassAlpha"),
    ('20', "Light Pass Additive", "kPCBlendModeLightPassAdditive"),
]

combiner_type_items = [
    ('0', "Default", "PC_CT_DEFAULT"),
    ('1', "Lightmap", "PC_CT_LIGHTMAP"),
    ('2', "Reflection", "PC_CT_REFLECTION"),
    ('3', "Masked Reflection", "PC_CT_MASKEDREFLECTION"),
    ('4', "Stencil Reflection", "PC_CT_STENCILREFLECTION"),
    ('5', "Diffuse", "PC_CT_DIFFUSE"),
    ('6', "Masked Diffuse", "PC_CT_MASKEDDIFFUSE"),
    ('7', "Immediate Draw", "PC_CT_IMMEDIATEDRAW"),
    ('8', "Immediate Draw Predator", "PC_CT_IMMEDIATEDRAW_PREDATOR"),
    ('9', "Depth of Field", "PC_CT_DEPTHOFFIELD"),
    ('10', "Count", "PC_CT_COUNT"),
]

class TR7AE_ModelTargetInfo(bpy.types.PropertyGroup):
    px: bpy.props.FloatProperty(name="Position X")
    py: bpy.props.FloatProperty(name="Position Y")
    pz: bpy.props.FloatProperty(name="Position Z")
    rx: bpy.props.FloatProperty(name="Rotation X")
    ry: bpy.props.FloatProperty(name="Rotation Y")
    rz: bpy.props.FloatProperty(name="Rotation Z")
    unique_id: bpy.props.IntProperty(name="Unique ID")
    flags: bpy.props.EnumProperty(
        name="Flags",
        items=TargetFlagItems,
        options={'ENUM_FLAG'}
    )

def update_backface_culling(self, context):
    mat = self.id_data
    if isinstance(mat, bpy.types.Material):
        mat.use_backface_culling = not self.double_sided

def update_specular_power(self, context):
    mat = self.id_data
    if isinstance(mat, bpy.types.Material) and mat.use_nodes:
        divide_node = mat.node_tree.nodes.get("SpecularPower_Divide")
        if divide_node and divide_node.type == 'MATH' and divide_node.operation == 'DIVIDE':
            divide_node.inputs[1].default_value = self.specular_power if self.specular_power != 0 else 1.0

class NextGenMaterialProperties(bpy.types.PropertyGroup):
    vertex_shader_flags: bpy.props.IntProperty(
        name="Vertex Shader Flags",
        description="LOREM IPSUM",
        default=0
    )

    mat_id: bpy.props.IntProperty(
        name="Material ID",
        description="LOREM IPSUM",
        default=1,
        min=0,
        max=2147483647
    )

    blend_mode: bpy.props.EnumProperty(
        name="",
        description="LOREM IPSUM",
        items=blend_mode_items,
        default='0'
    )

    combiner_type: bpy.props.EnumProperty(
        name="",
        description="LOREM IPSUM",
        items=combiner_type_items,
        default='0',
    )

    flags: bpy.props.IntProperty(
        name="Flags",
        description="LOREM IPSUM",
        default=10,
        min=0,
        max=2147483647
    )

    double_sided: bpy.props.BoolProperty(
        name="Double Sided",
        description="Whether the mesh has double-sided winding data",
        default=False,
        update=update_backface_culling  # Add this line
    )

    opacity: bpy.props.FloatProperty(
        name="Opacity",
        description="LOREM IPSUM",
        default=1.0,
        min=0.0,
        max=1.0
    )

    poly_flags: bpy.props.IntProperty(
        name="Polygon Flags",
        description="LOREM IPSUM",
        default=0,
        min=0,
        max=2147483647
    )

    uv_auto_scroll_speed: bpy.props.FloatProperty(
        name="UV Auto Scroll Speed",
        description="LOREM IPSUM",
        default=10.0,
        min=0.0,
        max=1000000.0
    )

    sort_bias: bpy.props.FloatProperty(
        name="Sort Bias",
        description="LOREM IPSUM",
        default=10.0,
        min=0.0,
        max=1000000.0
    )

    detail_range_mul: bpy.props.FloatProperty(
        name="Detail Range Multiplier",
        description="LOREM IPSUM",
        default=10.0,
        min=0.0,
        max=1000000.0
    )

    detail_scale: bpy.props.FloatProperty(
        name="Detail Scale",
        description="LOREM IPSUM",
        default=10.0,
        min=0.0,
        max=1000000.0
    )

    parallax_scale: bpy.props.FloatProperty(
        name="Parallax Scale",
        description="LOREM IPSUM",
        default=0.05,
        min=0.0,
        max=1000000.0
    )

    parallax_offset: bpy.props.FloatProperty(
        name="Parallax Offset",
        description="LOREM IPSUM",
        default=0.05,
        min=0.0,
        max=1000000.0
    )
    
    specular_power: bpy.props.FloatProperty(
        name="Specular Power",
        description="Controls the shininess of the material",
        default=50.0,
        min=0.01,
        max=512.0,
        update=update_specular_power
    )

    specular_shift_0: bpy.props.FloatProperty(
        name="Specular Shift 0",
        description="LOREM IPSUM",
        default=10.0,
        min=0.0,
        max=1000000.0
    )

    specular_shift_1: bpy.props.FloatProperty(
        name="Specular Shift 1",
        description="LOREM IPSUM",
        default=10.0,
        min=0.0,
        max=1000000.0
    )

    rim_light_color: bpy.props.FloatVectorProperty(
        name="Rim Light Color",
        description="LOREM IPSUM",
        size=4,
        subtype='COLOR',
        default=(1.0, 1.0, 1.0, 1.0),
        min=0.0,
        max=1.0
    )
    
    rim_light_intensity: bpy.props.FloatProperty(
        name="Rim Light Intensity",
        description="LOREM IPSUM",
        default=1.0,
        min=0.0,
        max=1000000.0
    )

    water_blend_bias: bpy.props.FloatProperty(
        name="Water Blend Bias",
        description="LOREM IPSUM",
        default=1.0,
        min=0.0,
        max=1000000.0
    )

    water_blend_exponent: bpy.props.FloatProperty(
        name="Water Blend Exponent",
        description="LOREM IPSUM",
        default=1.0,
        min=0.0,
        max=1000000.0
    )

    water_deep_color: bpy.props.FloatVectorProperty(
        name="Water Deep Color",
        description="LOREM IPSUM",
        size=4,
        subtype='COLOR',
        default=(1.0, 1.0, 1.0, 1.0),
        min=0.0,
        max=1.0
    )

class TR7AE_SphereInfo(bpy.types.PropertyGroup):
    flags: bpy.props.IntProperty(name="Flags")
    id: bpy.props.IntProperty(name="ID")
    rank: bpy.props.IntProperty(name="Rank")
    radius: bpy.props.FloatProperty(name="Radius")
    x: bpy.props.FloatProperty(name="X")
    y: bpy.props.FloatProperty(name="Y")
    z: bpy.props.FloatProperty(name="Z")
    radius_sq: bpy.props.FloatProperty(name="Radius Squared")
    mass: bpy.props.FloatProperty(name="Mass")
    buoyancy_factor: bpy.props.FloatProperty(name="Buoyancy Factor")
    explosion_factor: bpy.props.FloatProperty(name="Explosion Factor")
    material_type: bpy.props.IntProperty(name="Material Type")
    pad: bpy.props.IntProperty(name="Pad")
    damage: bpy.props.FloatProperty(name="Damage")

class TR7AE_HBoxInfo(bpy.types.PropertyGroup):
    widthx: bpy.props.FloatProperty(name="Width X")
    widthy: bpy.props.FloatProperty(name="Width Y")
    widthz: bpy.props.FloatProperty(name="Width Z")
    widthw: bpy.props.FloatProperty(name="Width W")
    positionboxx: bpy.props.FloatProperty(name="Pos X")
    positionboxy: bpy.props.FloatProperty(name="Pos Y")
    positionboxz: bpy.props.FloatProperty(name="Pos Z")
    positionboxw: bpy.props.FloatProperty(name="Pos W")
    quatx: bpy.props.FloatProperty(name="Quat X")
    quaty: bpy.props.FloatProperty(name="Quat Y")
    quatz: bpy.props.FloatProperty(name="Quat Z")
    quatw: bpy.props.FloatProperty(name="Quat W")
    flags: bpy.props.IntProperty(name="Flags")
    id: bpy.props.IntProperty(name="ID")
    rank: bpy.props.IntProperty(name="Rank")
    mass: bpy.props.IntProperty(name="Mass")
    buoyancy_factor: bpy.props.IntProperty(name="Buoyancy Factor")
    explosion_factor: bpy.props.IntProperty(name="Explosion Factor")
    material_type: bpy.props.IntProperty(name="Material Type")
    pad: bpy.props.IntProperty(name="Pad")
    damage: bpy.props.IntProperty(name="Damage")
    pad1: bpy.props.IntProperty(name="Pad1")

class TR7AE_HCapsuleInfo(bpy.types.PropertyGroup):
    posx: bpy.props.FloatProperty(name="Pos X")
    posy: bpy.props.FloatProperty(name="Pos Y")
    posz: bpy.props.FloatProperty(name="Pos Z")
    posw: bpy.props.FloatProperty(name="Pos W")
    quatx: bpy.props.FloatProperty(name="Quat X")
    quaty: bpy.props.FloatProperty(name="Quat Y")
    quatz: bpy.props.FloatProperty(name="Quat Z")
    quatw: bpy.props.FloatProperty(name="Quat W")
    flags: bpy.props.IntProperty(name="Flags")
    id: bpy.props.IntProperty(name="ID")
    rank: bpy.props.IntProperty(name="Rank")
    radius: bpy.props.FloatProperty(name="Radius")
    length: bpy.props.FloatProperty(name="Length")
    mass: bpy.props.IntProperty(name="Mass")
    buoyancy_factor: bpy.props.IntProperty(name="Buoyancy Factor")
    explosion_factor: bpy.props.IntProperty(name="Explosion Factor")
    material_type: bpy.props.IntProperty(name="Material Type")
    pad: bpy.props.IntProperty(name="Pad")
    damage: bpy.props.IntProperty(name="Damage")

class TR7AE_ClothSettings(bpy.types.PropertyGroup):
    gravity: bpy.props.FloatProperty(name="Gravity", precision=1)
    drag: bpy.props.FloatProperty(name="Drag", precision=2)
    wind_response: bpy.props.FloatProperty(name="Wind Response", precision=2)
    flags: bpy.props.IntProperty(name="Flags")

class ClothJointMapData(bpy.types.PropertyGroup):
    segment: bpy.props.IntProperty(name="Segment")
    flags: bpy.props.IntProperty(name="Flags")
    axis: bpy.props.IntProperty(name="Axis")
    joint_order: bpy.props.IntProperty(name="Joint Order")
    center: bpy.props.IntProperty(name="Center Index")
    points: bpy.props.IntVectorProperty(name="Connected Points", size=4)
    bounds: bpy.props.FloatVectorProperty(name="Bounds", size=6)

import struct

def ushort_to_float(u):
    return struct.unpack('f', struct.pack('I', u << 16))[0]

def get_gnc_data_offset(filepath):
    with open(filepath, "rb") as f:
        f.seek(0xC)
        packed_data = struct.unpack("<I", f.read(4))[0]
        num_relocations = (packed_data >> 8) & 0xFFFFFF
        return 24 + (num_relocations * 8)
    
import struct
import bpy
from mathutils import Vector
from pathlib import Path
from collections import defaultdict

def try_import_cloth(filepath, armature_obj):
    from mathutils import Vector
    from pathlib import Path
    import bpy
    import struct

    folder = Path(filepath).parent
    cloth_files = list(folder.glob("*.cloth"))
    if not cloth_files:
        print("[CLOTH] No cloth file found in folder.")
        return

    cloth_path = cloth_files[0]
    print(f"[CLOTH] Found cloth file: {cloth_path.name}")

    def parse_cloth_points_and_maps(path):
        with open(path, "rb") as f:
            assert f.read(4) == b'SECT'
            f.seek(0xC)
            packed_data = struct.unpack("<I", f.read(4))[0]
            num_relocs = (packed_data >> 8) & 0xFFFFFF
            section_info_size = 0x18 + num_relocs * 8

            def resolve_ptr(reloc_index):
                f.seek(0x18 + reloc_index * 8 + 4)
                ptr_offset = struct.unpack("<I", f.read(4))[0]
                f.seek(section_info_size + ptr_offset)
                return struct.unpack("<I", f.read(4))[0]

            f.seek(0x18 + 0 * 8 + 4)
            cloth_setup_ptr_offset = struct.unpack("<I", f.read(4))[0]
            cloth_setup_offset = section_info_size + cloth_setup_ptr_offset

            f.seek(cloth_setup_offset + 8)
            gravity = struct.unpack("<f", f.read(4))[0]
            drag = struct.unpack("<f", f.read(4))[0]
            wind_response = struct.unpack("<f", f.read(4))[0]
            flags = struct.unpack("<H", f.read(2))[0]
            positionPoint = struct.unpack("<H", f.read(2))[0]
            num_points = struct.unpack("<H", f.read(2))[0]
            num_maps = struct.unpack("<H", f.read(2))[0]
            num_dist_rules = struct.unpack("<H", f.read(2))[0]

            points_offset = resolve_ptr(1)
            maps_offset = resolve_ptr(2)
            dist_rules_offset = resolve_ptr(3) if num_dist_rules > 0 else None

            # ClothPoints
            cloth_points = []
            f.seek(section_info_size + points_offset)
            for _ in range(num_points):
                data = f.read(20)
                segment, flags, joint_order, up_to = struct.unpack("<4H", data[:8])
                x, y, z = struct.unpack("<3f", data[8:])
                cloth_points.append({
                    "segment": segment,
                    "flags": flags,
                    "joint_order": joint_order,
                    "up_to": up_to,
                    "pos": (x, y, z)
                })

            joint_maps = []
            f.seek(section_info_size + maps_offset)
            for _ in range(num_maps):
                segment, flags = struct.unpack("<HH", f.read(4))
                axis = struct.unpack("<B", f.read(1))[0]
                joint_order = struct.unpack("<B", f.read(1))[0]
                center = struct.unpack("<H", f.read(2))[0]
                points = struct.unpack("<4H", f.read(8))
                bounds = struct.unpack("<6f", f.read(24))
                joint_maps.append({
                    "segment": segment,
                    "flags": flags,
                    "axis": axis,
                    "joint_order": joint_order,
                    "center": center,
                    "points": points,
                    "bounds": bounds
                })

            # DistRules
            dist_rules = []
            if dist_rules_offset is not None:
                f.seek(section_info_size + dist_rules_offset)
                for _ in range(num_dist_rules):
                    pt0, pt1, flag0, flag1 = struct.unpack("<4H", f.read(8))
                    min_dist, max_dist = struct.unpack("<2f", f.read(8))
                    dist_rules.append({
                        "point0": pt0,
                        "point1": pt1,
                        "flag0": flag0,
                        "flag1": flag1,
                        "min": min_dist,
                        "max": max_dist
                    })

            return cloth_points, joint_maps, dist_rules, gravity, drag, wind_response, flags

    def create_clothpoint_mesh(cloth_points, joint_maps, dist_rules, armature_obj):
        from mathutils import Vector
        import bpy

        for obj in list(bpy.data.objects):
            if obj.name == "Cloth":
                bpy.data.objects.remove(obj, do_unlink=True)

        cloth_empty = bpy.data.objects.new("Cloth", None)
        cloth_empty.empty_display_type = 'SPHERE'
        cloth_empty.empty_display_size = 0.5
        bpy.context.collection.objects.link(cloth_empty)
        cloth_empty.parent = armature_obj
        cloth_empty.matrix_parent_inverse.identity()

        verts = []
        for i, pt in enumerate(cloth_points):
            bone_index = pt["segment"]
            bone_name = f"Bone_{bone_index}"
            if bone_name in armature_obj.pose.bones:
                bone = armature_obj.pose.bones[bone_name]
                local_offset = Vector(pt["pos"])
                world_pos = armature_obj.matrix_world @ (bone.matrix @ local_offset)
                world_pos *= 100.0
            else:
                print(f"[WARN] Bone '{bone_name}' not found for ClothPoint {i}")
                world_pos = Vector((0, 0, 0))
            verts.append(world_pos)

            # if pt["flags"] == 1:
                # bpy.ops.mesh.primitive_uv_sphere_add(radius=20, segments=16, ring_count=8, location=world_pos)
                # sphere_obj = bpy.context.active_object
                # sphere_obj.name = f"ClothCollision_{i}"
                # sphere_obj.parent = cloth_empty
                # sphere_obj.matrix_parent_inverse.identity()

                # sphere_obj.display_type = 'WIRE'
                # sphere_obj.show_in_front = True

                # vg = sphere_obj.vertex_groups.new(name=bone_name)
                # vg.add(range(len(sphere_obj.data.vertices)), 1.0, 'REPLACE')

                # arm_mod = sphere_obj.modifiers.new(name="Armature", type='ARMATURE')
                # arm_mod.object = armature_obj

        edges = []
        for jm in joint_maps:
            a, b = jm["points"][0], jm["points"][1]
            if a < len(verts) and b < len(verts) and a != b:
                edges.append((a, b))
        for rule in dist_rules:
            a, b = rule["point0"], rule["point1"]
            if a < len(verts) and b < len(verts) and a != b:
                edges.append((a, b))

        mesh = bpy.data.meshes.new("ClothStrip")
        mesh.from_pydata(verts, edges, [])
        mesh.update()

        obj = bpy.data.objects.new("ClothStrip", mesh)
        obj.display_type = 'WIRE'
        obj.show_wire = True
        bpy.context.collection.objects.link(obj)

        if hasattr(obj, "cloth_points"):
            obj.cloth_points.clear()

        for pt in cloth_points:
            item = obj.cloth_points.add()
            item.segment = pt["segment"]
            item.flags = pt["flags"]
            item.joint_order = pt["joint_order"]
            item.up_to = pt["up_to"]
            item.pos = pt["pos"]

        obj.tr7ae_cloth.gravity = gravity
        obj.tr7ae_cloth.drag = drag
        obj.tr7ae_cloth.wind_response = wind_response
        obj.tr7ae_cloth.flags = flags

        obj.parent = cloth_empty
        obj.matrix_parent_inverse.identity()

        bone_groups = {}
        for pt in cloth_points:
            bone_index = pt["segment"]
            bone_name = f"Bone_{bone_index}"
            if bone_name not in bone_groups:
                bone_groups[bone_name] = obj.vertex_groups.new(name=bone_name)

        for i, pt in enumerate(cloth_points):
            bone_index = pt["segment"]
            bone_name = f"Bone_{bone_index}"
            if bone_name in bone_groups:
                bone_groups[bone_name].add([i], 1.0, 'REPLACE')

        arm_mod = obj.modifiers.new(name="Armature", type='ARMATURE')
        arm_mod.object = armature_obj
        obj["cloth_points"] = [pt.copy() for pt in cloth_points]
        if hasattr(obj, "tr7ae_distrules"):
            obj.tr7ae_distrules.clear()

        for rule in dist_rules:
            item = obj.tr7ae_distrules.add()
            item.point0 = rule["point0"]
            item.point1 = rule["point1"]
            item.flag0 = rule["flag0"]
            item.flag1 = rule["flag1"]
            item.min = rule["min"]
            item.max = rule["max"]

        if hasattr(obj, "tr7ae_jointmaps"):
            obj.tr7ae_jointmaps.clear()

        for jm in joint_maps:
            item = obj.tr7ae_jointmaps.add()
            item.segment = jm["segment"]
            item.flags = jm["flags"]
            item.axis = jm["axis"]
            item.joint_order = jm["joint_order"]
            item.center = jm["center"]
            item.points = jm["points"]
            item.bounds = jm["bounds"]

        print(f"[CLOTH] Imported {len(verts)} cloth points, {len(joint_maps)} joint maps, {len(dist_rules)} dist rules with bone skinning and UV spheres for flagged points.")

    cloth_points, joint_maps, dist_rules, gravity, drag, wind_response, flags = parse_cloth_points_and_maps(cloth_path)
    create_clothpoint_mesh(cloth_points, joint_maps, dist_rules, armature_obj)


def import_envmapped_vertices(filepath, env_mapped_vertices, num_relocations):
    import struct, os

    section_index = None
    skip_extra_bytes = False

    with open(filepath, "rb") as f:
        f.seek(24)
        for _ in range(num_relocations):
            packed = int.from_bytes(f.read(2), "little")
            section_index_candidate = packed >> 3
            type_specific = int.from_bytes(f.read(2), "little")
            offset = int.from_bytes(f.read(4), "little")

            if offset == 92:
                section_index = section_index_candidate
                if type_specific == 25199:
                    skip_extra_bytes = True
                break

    if section_index is None:
        print("[INFO] No relocation with offset 92 found.")
        return []

    gnc_path = os.path.join(os.path.dirname(filepath), f"{section_index}_0.gnc")
    if not os.path.exists(gnc_path):
        print(f"[WARNING] GNC file not found: {gnc_path}")
        return []

    with open(gnc_path, "rb") as f:
        data_start = get_gnc_data_offset(gnc_path)
        f.seek(data_start + env_mapped_vertices)

        count = struct.unpack("<I", f.read(4))[0]
        data = f.read(count * 2)
        indices = list(struct.unpack(f"<{count}H", data))

    return indices

def import_eyerefenvmapped_vertices(filepath, eye_ref_env_mapped_vertices, num_relocations):
    import struct, os

    section_index = None
    skip_extra_bytes = False

    with open(filepath, "rb") as f:
        f.seek(24)
        for _ in range(num_relocations):
            packed = int.from_bytes(f.read(2), "little")
            section_index_candidate = packed >> 3
            type_specific = int.from_bytes(f.read(2), "little")
            offset = int.from_bytes(f.read(4), "little")

            if offset == 96:
                section_index = section_index_candidate
                if type_specific == 28527:
                    skip_extra_bytes = True
                break

    if section_index is None:
        print("[INFO] No relocation with offset 92 found.")
        return []

    gnc_path = os.path.join(os.path.dirname(filepath), f"{section_index}_0.gnc")
    if not os.path.exists(gnc_path):
        print(f"[WARNING] GNC file not found: {gnc_path}")
        return []

    with open(gnc_path, "rb") as f:
        data_start = get_gnc_data_offset(gnc_path)
        f.seek(data_start + eye_ref_env_mapped_vertices)

        count = struct.unpack("<I", f.read(4))[0]
        data = f.read(count * 2)
        indices = list(struct.unpack(f"<{count}H", data))

    return indices



def import_vertex_colors(filepath, vertex_color_offset, num_relocations, num_vertices):
    import struct, os

    section_index = None

    with open(filepath, "rb") as f:
        f.seek(24)

        for i in range(num_relocations):
            entry_start = f.tell()
            packed = int.from_bytes(f.read(2), "little")
            section_index_candidate = packed >> 3
            f.read(2)
            offset = int.from_bytes(f.read(4), "little")

            if offset == 100:
                section_index = section_index_candidate
                break

    if section_index is None:
        print("[INFO] No relocation with offset 100 found. Vertex colors skipped.")
        return None

    gnc_path = os.path.join(os.path.dirname(filepath), f"{section_index}_0.gnc")
    if not os.path.exists(gnc_path):
        print(f"[WARNING] GNC file not found: {gnc_path}")
        return None

    vertex_colors = []
    with open(gnc_path, "rb") as f:
        data_start = get_gnc_data_offset(gnc_path)
        f.seek(data_start + vertex_color_offset)

        for _ in range(num_vertices):
            data = f.read(4)
            if len(data) < 4:
                break
            r, g, b, a = struct.unpack("4B", data)
            vertex_colors.append((b / 127.0, g / 127.0, r / 127.0, a / 127.0))

    return vertex_colors

from bpy.types import Operator, Panel
from bpy.props import StringProperty, IntProperty
from bpy_extras.io_utils import ImportHelper
from mathutils import Vector
import os
import bpy.utils

texture_dir = os.path.join(
    bpy.utils.user_resource('SCRIPTS'),
    "addons",
    "trlau_textures"
)
os.makedirs(texture_dir, exist_ok=True)


import struct
from pathlib import Path

def convert_pcd_to_dds(pcd_path, texture_dir):
    pcd_path = Path(pcd_path)
    dds_path = Path(texture_dir) / f"{pcd_path.stem}.dds"

    if os.path.exists(dds_path):
        print(f"DDS already exists: {os.path.basename(dds_path)}")
        return dds_path, None

    try:
        with open(pcd_path, "rb") as f:
            f.seek(28)
            format_bytes = f.read(4)
            dds_format = struct.unpack('<I', format_bytes)[0]

            f.seek(24)
            texture_header = f.read(24)
            (
                magic_number,
                _,
                bitmap_size,
                _,
                width,
                height,
                _,
                mipmaps,
                _
            ) = struct.unpack("<I i I I H H B B H", texture_header)

            if magic_number != 0x39444350:
                raise ValueError(f"Unexpected magic number: {hex(magic_number)}")

            f.seek(48)
            dxt_data = f.read(bitmap_size)

            def create_dds_header(width, height, mipmaps, dds_format, dxt_size):
                header = bytearray(128)
                struct.pack_into('<4sI', header, 0, b'DDS ', 124)
                struct.pack_into('<I', header, 8, 0x0002100F)
                struct.pack_into('<I', header, 12, height)
                struct.pack_into('<I', header, 16, width)
                struct.pack_into('<I', header, 20, max(1, dxt_size))
                struct.pack_into('<I', header, 28, mipmaps if mipmaps > 0 else 1)

                if dds_format == 21:
                    struct.pack_into('<I', header, 76, 32)
                    struct.pack_into('<I', header, 80, 0x00000041)
                    struct.pack_into('<4s', header, 84, b'\x00\x00\x00\x00')
                    struct.pack_into('<I', header, 88, 32)
                    struct.pack_into('<I', header, 92, 0x00FF0000)
                    struct.pack_into('<I', header, 96, 0x0000FF00)
                    struct.pack_into('<I', header, 100, 0x000000FF)
                    struct.pack_into('<I', header, 104, 0xFF000000)
                else:
                    struct.pack_into('<I', header, 76, 32)
                    struct.pack_into('<I', header, 80, 0x00000004)
                    struct.pack_into('<4s', header, 84, struct.pack("<I", dds_format))

                struct.pack_into('<I', header, 108, 0x1000)
                return header

            dds_header = create_dds_header(width, height, mipmaps, dds_format, len(dxt_data))

            with open(dds_path, "wb") as out:
                out.write(dds_header)
                out.write(dxt_data)

            print(f"Converted {pcd_path.name} → {dds_path.name} ({width}x{height}, mipmaps: {mipmaps})")
            return dds_path, dds_format

    except Exception as e:
        print(f"[ERROR] DDS conversion failed: {e}")
        return None


class TR7AE_OT_ImportModel(Operator, ImportHelper):
    bl_idname = "tr7ae.import_model"
    bl_label = "Import TR7AE Model"
    bl_description = "Import a Tomb Raider Legend/Anniversary Model"
    bl_options = {'REGISTER', 'UNDO'}

    filename_ext = ".tr7aemesh"
    filter_glob: StringProperty(default="*.tr7aemesh", options={'HIDDEN'})

    import_textures: bpy.props.BoolProperty(
        name="Import Textures",
        description="Import and assign textures",
        default=True
    )

    import_cloth: bpy.props.BoolProperty(
        name="Import Cloth",
        description="Import cloth physics",
        default=True
    )

    import_hinfo: bpy.props.BoolProperty(
        name="Import HInfo",
        description="Import HSpheres, HBoxes, HMarkers and HCapsules where applicable",
        default=True
    )


    def draw(self, context):
        layout = self.layout
        layout.prop(self, "import_textures")
        layout.prop(self, "import_cloth")
        layout.prop(self, "import_hinfo")

    def execute(self, context):
        filepath = self.filepath
        try:
            with open(filepath, 'rb') as f:
                self.import_tr7ae(f, context, filepath, self.import_cloth, self.import_hinfo, self.import_textures)
        except Exception as e:
            self.report({'ERROR'}, f"Failed to import model: {e}")
            return {'CANCELLED'}
        return {'FINISHED'}

    def import_tr7ae(self, fhandle, context, filepath=None, do_import_cloth=True, do_import_hinfo=True, do_import_textures=True):
        converted_textures = {}
        loaded_images = {}
        def read(fmt):
            size = struct.calcsize(fmt)
            data = fhandle.read(size)
            if len(data) != size:
                raise ValueError("Unexpected EOF")
            return struct.unpack(fmt, data)

        def read_string(length):
            data = fhandle.read(length)
            if len(data) != length:
                raise ValueError("Unexpected EOF while reading string")
            return data.decode('ascii')

        fhandle.seek(0)
        if read_string(4) != "SECT":
            raise ValueError("Invalid file!\n\nRemoved Section header?")

        fhandle.seek(0xC)
        packed_data = read("<I")[0]
        num_relocations = (packed_data >> 8) & 0xFFFFFF
        section_info_size = num_relocations * 8 + 0x18

        fhandle.seek(section_info_size)
        version, num_bones, num_virtsegments, bone_data_offset = read("<4I")
        scale_vals = read("<4f")

        num_vertices, vertex_list = read("<iI")
        num_normals, normal_list = read("<iI")
        num_faces = read("<i")[0]
        mface_list = read("<I")[0]
        OBSOLETEaniTextures = read("<I")[0]
        max_rad, max_rad_sq = read("<2f")
        OBSOLETEstartTextures = read("<I")[0]
        OBSOLETEendTextures = read("<I")[0]
        animatedListInfo = read("<I")[0]
        animatedInfo = read("<I")[0]
        scrollInfo = read("<I")[0]
        face_list = read("<I")[0]
        env_mapped_vertices = read("<I")[0]
        envmapped_indices = import_envmapped_vertices(filepath, env_mapped_vertices, num_relocations)
        eye_ref_env_mapped_vertices = read("<I")[0]
        eyerefenvmapped_indices = import_eyerefenvmapped_vertices(filepath, eye_ref_env_mapped_vertices, num_relocations)
        material_vertex_colors_offset = read("<I")[0]
        vertex_colors = import_vertex_colors(filepath, material_vertex_colors_offset, num_relocations, num_vertices)
        spectralVertexColors = read("<I")[0]
        pnShadowFaces = read("<I")[0]
        pnShadowEdges = read("<I")[0]
        bone_mirror_data_offset = read("<I")[0]
        drawgroupCenterList = read("<I")[0]
        numMarkUps = read("<I")[0]
        markUpList = read("<I")[0]
        markups = import_markups(filepath, markUpList, numMarkUps, num_relocations, is_legend(filepath))
        numTargets = read("<I")[0]
        targetList = read("<I")[0]
        model_targets = import_model_targets(filepath, targetList, numTargets, num_relocations)
        cdcRenderDataID = read("<I")[0]
        if bone_mirror_data_offset > 0:
            fhandle.seek(bone_mirror_data_offset + section_info_size)
            bone_mirror_data = []
            while True:
                current_pos = fhandle.tell()
                check_zero = struct.unpack('<H', fhandle.read(2))[0]
                fhandle.seek(current_pos)
                if check_zero == 0:
                    break
                bone1, bone2, count = struct.unpack('<3B', fhandle.read(3))
                bone_mirror_data.append({'bone1': bone1, 'bone2': bone2, 'count': count})

        fhandle.seek(bone_data_offset + section_info_size)
        def read_hinfo(file, hinfo_offset, section_info_size):
            if hinfo_offset == 0:
                return None

            current_pos = file.tell()
            try:
                file.seek(hinfo_offset + section_info_size)
                hinfo = {}

                num_hspheres = struct.unpack("<i", file.read(4))[0]
                hsphere_list = struct.unpack("<I", file.read(4))[0]
                num_hboxes = struct.unpack("<i", file.read(4))[0]
                hbox_list = struct.unpack("<I", file.read(4))[0]
                num_hmarkers = struct.unpack("<i", file.read(4))[0]
                hmarker_list = struct.unpack("<I", file.read(4))[0]
                num_hcapsules = struct.unpack("<i", file.read(4))[0]
                hcapsule_list = struct.unpack("<I", file.read(4))[0]

                if num_hspheres > 0 and hsphere_list != 0:
                    file.seek(hsphere_list + section_info_size)
                    spheres = []
                    for _ in range(num_hspheres):
                        flags = struct.unpack("<H", file.read(2))[0]
                        id = struct.unpack("<B", file.read(1))[0]
                        rank = struct.unpack("<B", file.read(1))[0]
                        radius = struct.unpack("<H", file.read(2))[0]
                        x = struct.unpack("<h", file.read(2))[0]
                        y = struct.unpack("<h", file.read(2))[0]
                        z = struct.unpack("<h", file.read(2))[0]
                        radius_sq = struct.unpack("<I", file.read(4))[0]
                        mass = struct.unpack("<H", file.read(2))[0]
                        buoyancy_factor = struct.unpack("<B", file.read(1))[0]
                        explosion_factor = struct.unpack("<B", file.read(1))[0]
                        material_type = struct.unpack("<B", file.read(1))[0]
                        pad = struct.unpack("<B", file.read(1))[0]
                        damage = struct.unpack("<H", file.read(2))[0]
                        spheres.append({
                            "flags": flags,
                            "id": id,
                            "rank": rank,
                            "radius": radius,
                            "x": x,
                            "y": y,
                            "z": z,
                            "radius_sq": radius_sq,
                            "mass": mass,
                            "buoyancy_factor": buoyancy_factor,
                            "explosion_factor": explosion_factor,
                            "material_type": material_type,
                            "pad": pad,
                            "damage": damage,
                        })
                    hinfo["spheres"] = spheres

                if num_hboxes > 0 and hbox_list != 0:
                    file.seek(hbox_list + section_info_size)
                    boxes = []
                    for _ in range(num_hboxes):
                        wx, wy, wz, ww = struct.unpack("<4f", file.read(16))
                        positionboxx, positionboxy, positionboxz, positionboxw = struct.unpack("<4f", file.read(16))
                        qx, qy, qz, qw = struct.unpack("<4f", file.read(16))
                        flags    = struct.unpack("<h", file.read(2))[0]
                        id_      = struct.unpack("<B", file.read(1))[0]
                        rank     = struct.unpack("<B", file.read(1))[0]
                        mass     = struct.unpack("<H", file.read(2))[0]
                        buoyancy = struct.unpack("<B", file.read(1))[0]
                        expl     = struct.unpack("<B", file.read(1))[0]
                        mat_type = struct.unpack("<B", file.read(1))[0]
                        pad      = struct.unpack("<B", file.read(1))[0]
                        damage   = struct.unpack("<h", file.read(2))[0]
                        pad1     = struct.unpack("<I", file.read(4))[0]

                        boxes.append({
                            "widthx": wx, "widthy": wy, "widthz": wz, "widthw": ww,
                            "positionboxx": positionboxx,   "positionboxy": positionboxy,   "positionboxz": positionboxz,   "positionboxw": positionboxw,
                            "quatx": qx,  "quaty": qy,  "quatz": qz,  "quatw": qw,
                            "flags": flags,     "id": id_,       "rank": rank,
                            "mass": mass,       "buoyancy_factor": buoyancy,
                            "explosion_factor": expl,
                            "material_type": mat_type,
                            "pad": pad,         "damage": damage,
                            "pad1": pad1
                        })
                    hinfo["boxes"] = boxes


                if num_hcapsules > 0 and hcapsule_list != 0:
                    file.seek(hcapsule_list + section_info_size)
                    capsules = []
                    for _ in range(num_hcapsules):
                        posx  = struct.unpack("<f", file.read(4))[0]
                        posy  = struct.unpack("<f", file.read(4))[0]
                        posz  = struct.unpack("<f", file.read(4))[0]
                        posw  = struct.unpack("<f", file.read(4))[0]
                        quatx = struct.unpack("<f", file.read(4))[0]
                        quaty = struct.unpack("<f", file.read(4))[0]
                        quatz = struct.unpack("<f", file.read(4))[0]
                        quatw = struct.unpack("<f", file.read(4))[0]
                        flags = struct.unpack("<h", file.read(2))[0]
                        cid   = struct.unpack("<B", file.read(1))[0]
                        rank  = struct.unpack("<B", file.read(1))[0]
                        radius= struct.unpack("<H", file.read(2))[0]
                        length= struct.unpack("<H", file.read(2))[0]
                        mass  = struct.unpack("<H", file.read(2))[0]
                        buoyancy = struct.unpack("<B", file.read(1))[0]
                        explosion= struct.unpack("<B", file.read(1))[0]
                        mat_type = struct.unpack("<B", file.read(1))[0]
                        pad      = struct.unpack("<B", file.read(1))[0]
                        damage   = struct.unpack("<h", file.read(2))[0]

                        capsules.append({
                            "posx":            posx,
                            "posy":            posy,
                            "posz":            posz,
                            "posw":            posw,
                            "quatx":           quatx,
                            "quaty":           quaty,
                            "quatz":           quatz,
                            "quatw":           quatw,
                            "flags":           flags,
                            "id":              cid,
                            "rank":            rank,
                            "radius":          radius,
                            "length":          length,
                            "mass":            mass,
                            "buoyancy_factor": buoyancy,
                            "explosion_factor":explosion,
                            "material_type":   mat_type,
                            "pad":             pad,
                            "damage":          damage,
                        })
                    hinfo["capsules"] = capsules

                if num_hmarkers > 0 and hmarker_list != 0:
                    file.seek(hmarker_list + section_info_size)
                    markers = []
                    for _ in range(num_hmarkers):
                        bone = struct.unpack("<i", file.read(4))[0]
                        index = struct.unpack("<i", file.read(4))[0]
                        px, py, pz = struct.unpack("<3f", file.read(12))
                        rx, ry, rz = struct.unpack("<3f", file.read(12))
                        markers.append({
                            "bone": bone, "index": index,
                            "pos": (px, py, pz),
                            "rot": (rx, ry, rz)
                        })
                    hinfo["markers"] = markers

                return hinfo
            finally:
                file.seek(current_pos)

        bones, world_positions = [], {}
        for i in range(num_bones):
            min_q, max_q, pivot_q = read("<4f"), read("<4f"), read("<4f")
            flags, first_vert, last_vert, parent, hInfo = read("<i"), read("<h"), read("<h"), read("<i"), read("<I")
            pivot = Vector(pivot_q[:3])
            bone_data = {
                "pivot": pivot,
                "parent": parent[0],
                "min": min_q,
                "max": max_q,
                "flags": flags[0],
                "range": (first_vert[0], last_vert[0]),
                "hInfo": hInfo[0],
            }

            if do_import_hinfo:
                hinfo_struct = read_hinfo(fhandle, hInfo[0], section_info_size)
                bone_data["hinfo"] = hinfo_struct

            bones.append(bone_data)
            parent_idx = parent[0]
            world_positions[i] = pivot if parent_idx < 0 else world_positions[parent_idx] + pivot

        from mathutils import Matrix

        bone_matrices = {}
        for i, bone in enumerate(bones):
            trans = Matrix.Translation(world_positions[i])
            bone_matrices[i] = trans.to_3x3()

        fhandle.seek(bone_data_offset + section_info_size + num_bones * 64)
        virtsegment_map, virtsegment_data = {}, []
        for _ in range(num_virtsegments):
            read("<4f")
            read("<4f")
            read("<4f")
            flags = read("<i")[0]
            first_vertex, last_vertex, index, weight_index = read("<4h")
            weight = read("<f")[0]
            virtsegment_data.append((first_vertex, last_vertex, index, weight_index, weight))
            for v in range(first_vertex, last_vertex + 1):
                virtsegment_map[v] = index


        fhandle.seek(vertex_list + section_info_size)
        verts, uvs, normals, segments = [], [], [], []
        for i in range(num_vertices):
            vx, vy, vz = read("<3h")
            nx, ny, nz = struct.unpack("<3b", fhandle.read(3))
            pad = read("<b")[0]
            segment = read("<h")[0]
            uvx_raw, uvy_raw = read("<HH")
            
            uvx = ushort_to_float(uvx_raw)
            uvy = 1.0 - ushort_to_float(uvy_raw)

            uvs.append((uvx, uvy))
            
            base_pos = Vector((vx * scale_vals[0], vy * scale_vals[1], vz * scale_vals[2]))
            bone_index = virtsegment_map.get(i, segment)
            if 0 <= bone_index < len(bones):
                base_pos += world_positions[bone_index]
            verts.append(base_pos)

            # Apply normal rotation after bone_index is determined
            raw_normal = Vector((nx / 127.0, ny / 127.0, nz / 127.0))
            if 0 <= bone_index < len(bones):
                bone_matrix = bone_matrices[bone_index]
                transformed_normal = raw_normal @ bone_matrix
                transformed_normal.normalize()
            else:
                transformed_normal = raw_normal.normalized()
            normals.append(transformed_normal)

            segments.append(segment)

        fhandle.seek(face_list + section_info_size)
        mesh_chunks = []
        while True:
            start = fhandle.tell()
            preview = fhandle.read(2)
            if len(preview) < 2 or struct.unpack("<h", preview)[0] == 0:
                break
            fhandle.seek(start)
            vertex_count, draw_group = read("<2h")
            tpageid = struct.unpack("<I", fhandle.read(4))[0]
            texture_id        =  tpageid        & 0x1FFF
            blend_value       = (tpageid >> 13) & 0xF
            unknown_1         = (tpageid >> 17) & 0x7
            unknown_2         = (tpageid >> 20) & 0x1
            single_sided      = (tpageid >> 21) & 0x1
            texture_wrap      = (tpageid >> 22) & 0x3
            unknown_3         = (tpageid >> 24) & 0x1
            unknown_4         = (tpageid >> 25) & 0x1
            flat_shading      = (tpageid >> 26) & 0x1
            sort_z            = (tpageid >> 27) & 0x1
            stencil_pass      = (tpageid >> 28) & 0x3
            stencil_func      = (tpageid >> 30) & 0x1
            alpha_ref         = (tpageid >> 31) & 0x1
            fhandle.read(12)
            fhandle.seek(-4, 1)
            next_texture = read("<I")[0]
            face_data = fhandle.read(vertex_count * 2)
            indices = struct.unpack(f"<{vertex_count}H", face_data)
            chunk_faces, chunk_verts_set = [], set()
            for i in range(0, len(indices), 3):
                if i + 2 < len(indices):
                    tri = (indices[i], indices[i + 1], indices[i + 2])
                    chunk_faces.append(tri)
                    chunk_verts_set.update(tri)
            chunk_vert_map = {v: idx for idx, v in enumerate(sorted(chunk_verts_set))}
            remapped_faces = [tuple(chunk_vert_map[i] for i in face) for face in chunk_faces]
            chunk_verts = [verts[i] for i in sorted(chunk_verts_set)]
            chunk_uvs = [uvs[i] for i in sorted(chunk_verts_set)]
            chunk_normals = [normals[i] for i in sorted(chunk_verts_set)]
            chunk_segments = [segments[i] for i in sorted(chunk_verts_set)]
            mesh_chunks.append((chunk_verts, remapped_faces, chunk_uvs, chunk_normals, chunk_segments, draw_group, chunk_vert_map, texture_id, blend_value, unknown_1, unknown_2, single_sided, texture_wrap, unknown_3, unknown_4, flat_shading, sort_z, stencil_pass, stencil_func, alpha_ref))
            if next_texture == 0:
                break
            fhandle.seek(next_texture + section_info_size)

        armature_data = bpy.data.armatures.new("Armature")
        armature_obj = bpy.data.objects.new("Armature", armature_data)
        context.collection.objects.link(armature_obj)
        armature_obj["max_rad"] = max_rad
        armature_obj["cdcRenderDataID"] = cdcRenderDataID
        if bone_mirror_data_offset > 0:
            armature_obj["bone_mirror_data"] = bone_mirror_data
        armature_obj.scale = (0.01, 0.01, 0.01)
        bpy.context.view_layer.update()
        bpy.context.view_layer.objects.active = armature_obj
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

        from pathlib import Path
        import os

        main_file_index = int(Path(filepath).stem.split("_")[0])
        armature_obj.tr7ae_sections.main_file_index = main_file_index

        with open(filepath, "rb") as f:
            f.seek(24)
            for _ in range(num_relocations):
                packed = int.from_bytes(f.read(2), "little")
                section_index_candidate = packed >> 3
                f.read(2)
                offset = int.from_bytes(f.read(4), "little")
                if offset == 100:
                    armature_obj.tr7ae_sections.extra_file_index = section_index_candidate
                    break

        folder = Path(filepath).parent
        for file in folder.glob("*.cloth"):
            with open(file, "rb") as cf:
                cf.seek(24)
                packed = int.from_bytes(cf.read(2), "little")
                cloth_index = packed >> 3
                armature_obj.tr7ae_sections.cloth_file_index = cloth_index
                print(f"[INFO] Cloth section index = {cloth_index}")
                break

        armature_obj.show_in_front = True
        armature_obj.data.display_type = 'STICK'

        bpy.context.view_layer.objects.active = armature_obj
        bpy.ops.object.mode_set(mode='EDIT')

        bone_objects = []
        for i, bone in enumerate(bones):
            b = armature_data.edit_bones.new(f"Bone_{i}")
            b.head = world_positions[i]
            b.tail = world_positions[i] + Vector((0, 0.05, 0))
            bone_objects.append(b)
        for i, bone in enumerate(bones):
            parent_idx = bone["parent"]
            if parent_idx >= 0:
                bone_objects[i].parent = bone_objects[parent_idx]
        bpy.ops.object.mode_set(mode='OBJECT')

        global_marker_index = 0
        global_sphere_index = 0
        global_box_index = 0
        global_capsule_index = 0
        for i, bone in enumerate(bones):
            bone_name = f"Bone_{i}"
            pbone = armature_obj.pose.bones.get(bone_name)
            if pbone:
                pbone["flags"] = bone["flags"]
                pbone["min_q"] = bone["min"]
                pbone["max_q"] = bone["max"]
            hinfo = bone.get("hinfo")

            if hinfo:
                if "spheres" in hinfo:
                    clean_spheres = []
                    for sphere in hinfo["spheres"]:
                        clean_sphere = {
                            "flags": int(sphere["flags"]),
                            "id": int(sphere["id"]),
                            "rank": int(sphere["rank"]),
                            "radius": int(sphere["radius"]),
                            "x": int(sphere["x"]),
                            "y": int(sphere["y"]),
                            "z": int(sphere["z"]),
                            "radius_sq": int(sphere["radius_sq"]),
                            "mass": int(sphere["mass"]),
                            "buoyancy_factor": int(sphere["buoyancy_factor"]),
                            "explosion_factor": int(sphere["explosion_factor"]),
                            "material_type": int(sphere["material_type"]),
                            "pad": int(sphere["pad"]),
                            "damage": int(sphere["damage"]),
                        }
                        clean_spheres.append(clean_sphere)
                    pbone.tr7ae_hspheres.clear()
                    for sphere in clean_spheres:
                        item = pbone.tr7ae_hspheres.add()
                        item.flags = sphere["flags"]
                        item.id = sphere["id"]
                        item.rank = sphere["rank"]
                        item.radius = sphere["radius"]
                        item.x = sphere["x"]
                        item.y = sphere["y"]
                        item.z = sphere["z"]
                        item.radius_sq = sphere["radius_sq"]
                        item.mass = sphere["mass"]
                        item.buoyancy_factor = sphere["buoyancy_factor"]
                        item.explosion_factor = sphere["explosion_factor"]
                        item.material_type = sphere["material_type"]
                        item.pad = sphere["pad"]
                        item.damage = sphere["damage"]

                if "boxes" in hinfo:
                    pbone.tr7ae_hboxes.clear()
                    for box in hinfo["boxes"]:
                        item = pbone.tr7ae_hboxes.add()
                        item.widthx = box["widthx"]
                        item.widthy = box["widthy"]
                        item.widthz = box["widthz"]
                        item.widthw = box["widthw"]
                        item.positionboxx   = box["positionboxx"]
                        item.positionboxy   = box["positionboxy"]
                        item.positionboxz   = box["positionboxz"]
                        item.positionboxw   = box["positionboxw"]
                        item.quatx  = box["quatx"]
                        item.quaty  = box["quaty"]
                        item.quatz  = box["quatz"]
                        item.quatw  = box["quatw"]
                        item.flags  = box.get("flags", 0)
                        item.id     = box.get("id", 0)
                        item.rank   = box.get("rank", 0)
                        item.mass             = box.get("mass", 0)
                        item.buoyancy_factor  = box.get("buoyancy_factor", 0)
                        item.explosion_factor = box.get("explosion_factor", 0)
                        item.material_type    = box.get("material_type", 0)
                        item.pad              = box.get("pad", 0)
                        item.damage           = box.get("damage", 0)
                        item.pad1             = box.get("pad1", 0)

                if "capsules" in hinfo:
                    pbone.tr7ae_hcapsules.clear()

                    for cap in hinfo["capsules"]:
                        item = pbone.tr7ae_hcapsules.add()

                        item.posx = cap.get("posx", 0.0)
                        item.posy = cap.get("posy", 0.0)
                        item.posz = cap.get("posz", 0.0)
                        item.posw = cap.get("posw", 1.0)

                        item.quatx = cap.get("quatx", 0.0)
                        item.quaty = cap.get("quaty", 0.0)
                        item.quatz = cap.get("quatz", 0.0)
                        item.quatw = cap.get("quatw", 1.0)

                        item.flags = cap.get("flags", 0)
                        item.id    = cap.get("id",    0)
                        item.rank  = cap.get("rank",  0)

                        item.radius = cap.get("radius", 0)
                        item.length = cap.get("length", 0)

                        item.mass            = cap.get("mass",            0)
                        item.buoyancy_factor = cap.get("buoyancy_factor", 0)
                        item.explosion_factor= cap.get("explosion_factor",0)
                        item.material_type   = cap.get("material_type",   0)
                        item.pad             = cap.get("pad",             0)
                        item.damage          = cap.get("damage",          0)

                if "markers" in hinfo:
                    clean_markers = []
                    for marker in hinfo["markers"]:
                        clean_marker = {
                            "bone": int(marker["bone"]),
                            "index": int(marker["index"]),
                            "pos": list(map(float, marker["pos"])),
                            "rot": list(map(float, marker["rot"])),
                        }
                        clean_markers.append(clean_marker)
                    pbone.tr7ae_hmarkers.clear()
                    for marker in clean_markers:
                        item = pbone.tr7ae_hmarkers.add()
                        item.bone = marker["bone"]
                        item.index = marker["index"]
                        item.px, item.py, item.pz = marker["pos"]
                        item.rx, item.ry, item.rz = marker["rot"]

                from math import radians

                if "HInfo" not in bpy.data.objects:
                    hinfo_obj = bpy.data.objects.new("HInfo", None)
                    hinfo_obj.empty_display_type = 'PLAIN_AXES'
                    hinfo_obj.empty_display_size = 0.3
                    context.collection.objects.link(hinfo_obj)

                    hinfo_obj.parent = armature_obj
                    hinfo_obj.matrix_parent_inverse.identity()
                else:
                    hinfo_obj = bpy.data.objects["HInfo"]

                hinfo_collection = hinfo_obj.users_collection[0]

                if "markers" in hinfo:
                    for marker in hinfo["markers"]:
                        marker_name = f"HMarker_{global_marker_index}"
                        global_marker_index += 1
                        bone_name = f"Bone_{i}"

                        bpy.ops.mesh.primitive_cone_add(radius1=0.02, depth=0.05, vertices=8)
                        marker_obj = bpy.context.active_object
                        marker_obj.name = marker_name
                        marker_obj.scale = (100.0, 100.0, 100.0)
                        marker_obj["tr7ae_type"] = "HMarker"
                        marker_obj["tr7ae_bone_index"] = i

                        bone_matrix = armature_obj.matrix_world @ armature_obj.data.bones[bone_name].matrix_local
                        local_pos = Vector(marker['pos']).to_4d()
                        world_pos = bone_matrix @ local_pos
                        marker_obj.location = hinfo_obj.matrix_world.inverted() @ world_pos.to_3d()


                        rx, ry, rz = marker['rot']

                        marker_obj.rotation_mode = 'ZYX'
                        marker_obj.rotation_euler = (rx, ry, rz)


                        marker_obj.parent = hinfo_obj
                        marker_obj.matrix_parent_inverse.identity()

                        if marker_obj.name not in hinfo_collection.objects:
                            hinfo_collection.objects.link(marker_obj)

                        vg = marker_obj.vertex_groups.new(name=bone_name)
                        bpy.context.view_layer.objects.active = marker_obj
                        bpy.ops.object.mode_set(mode='EDIT')
                        bpy.ops.mesh.select_all(action='SELECT')
                        bpy.ops.object.mode_set(mode='OBJECT')
                        vg.add(range(len(marker_obj.data.vertices)), 1.0, 'REPLACE')

                        arm_mod = marker_obj.modifiers.new(name="Armature", type='ARMATURE')
                        arm_mod.object = armature_obj

                if "spheres" in hinfo:
                    for j, sphere in enumerate(hinfo["spheres"]):
                        sphere_name = f"HSphere_{global_sphere_index}"
                        global_sphere_index += 1
                        bone_name = f"Bone_{i}"

                        bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, segments=16, ring_count=8)
                        sphere_obj = bpy.context.active_object
                        sphere_obj.name = sphere_name

                        sphere_obj["tr7ae_type"]       = "HSphere"
                        sphere_obj["tr7ae_bone_index"] = i

                        r = sphere["radius"]
                        sphere_obj.scale = (r, r, r)

                        sphere_obj.display_type = 'WIRE'
                        sphere_obj.show_in_front = True


                        bone_matrix = armature_obj.matrix_world @ armature_obj.data.bones[bone_name].matrix_local
                        local_pos = Vector((sphere["x"], sphere["y"], sphere["z"])).to_4d()
                        world_pos = bone_matrix @ local_pos
                        sphere_obj.location = hinfo_obj.matrix_world.inverted() @ world_pos.to_3d()

                        sphere_obj.parent = hinfo_obj
                        sphere_obj.matrix_parent_inverse.identity()

                        if sphere_obj.name not in hinfo_collection.objects:
                            hinfo_collection.objects.link(sphere_obj)

                        vg = sphere_obj.vertex_groups.new(name=bone_name)
                        bpy.context.view_layer.objects.active = sphere_obj
                        bpy.ops.object.mode_set(mode='EDIT')
                        bpy.ops.mesh.select_all(action='SELECT')
                        bpy.ops.object.mode_set(mode='OBJECT')
                        vg.add(range(len(sphere_obj.data.vertices)), 1.0, 'REPLACE')

                        arm_mod = sphere_obj.modifiers.new(name="Armature", type='ARMATURE')
                        arm_mod.object = armature_obj

                if "boxes" in hinfo:
                    for j, box in enumerate(hinfo["boxes"]):
                        box_name  = f"HBox_{global_box_index}"
                        global_box_index += 1
                        bone_name = f"Bone_{i}"

                        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 0))
                        box_obj = bpy.context.active_object
                        box_obj.name = box_name

                        box_obj.show_in_front = True
                        box_obj.display_type  = 'WIRE'

                        pos_vec = Vector((
                            box["positionboxx"],
                            box["positionboxy"],
                            box["positionboxz"],
                        ))
                        quat = Quaternion((
                            box["quatw"],
                            box["quatx"],
                            box["quaty"],
                            box["quatz"],
                        )).normalized()

                        scale_mat = Matrix.Diagonal((
                            box["widthx"],
                            box["widthy"],
                            box["widthz"],
                            1.0
                        ))

                        local_box_mat = Matrix.Translation(pos_vec) @ quat.to_matrix().to_4x4() @ scale_mat

                        bone_mat = armature_obj.matrix_world @ armature_obj.data.bones[bone_name].matrix_local
                        world_box_mat = bone_mat @ local_box_mat
                        box_obj.matrix_world = hinfo_obj.matrix_world.inverted() @ world_box_mat

                        box_obj.parent                  = hinfo_obj
                        box_obj.parent_type             = 'OBJECT'
                        box_obj.matrix_parent_inverse.identity()

                        if box_obj.name not in hinfo_collection.objects:
                            hinfo_collection.objects.link(box_obj)
                            for coll in box_obj.users_collection:
                                if coll is not hinfo_collection:
                                    coll.objects.unlink(box_obj)

                        box_obj.parent      = hinfo_obj
                        box_obj.parent_type = 'OBJECT'

                        if box_obj.name not in hinfo_collection.objects:
                            hinfo_collection.objects.link(box_obj)
                            for coll in box_obj.users_collection:
                                if coll is not hinfo_collection:
                                    coll.objects.unlink(box_obj)

                        arm_mod = box_obj.modifiers.new(name="Armature", type='ARMATURE')
                        arm_mod.object = armature_obj

                        vg = box_obj.vertex_groups.new(name=bone_name)
                        vg.add([v.index for v in box_obj.data.vertices], 1.0, 'REPLACE')

                        box_obj["tr7ae_type"]       = "HBox"
                        box_obj["tr7ae_bone_index"] = i
                        box_obj["tr7ae_damage"]     = box["damage"]

                if "capsules" in hinfo:
                    for j, cap in enumerate(hinfo["capsules"]):
                        capsule_name = f"HCapsule_{global_capsule_index}"
                        global_capsule_index += 1
                        bone_name = f"Bone_{i}"

                        bpy.ops.mesh.primitive_cylinder_add(
                            radius=1.0,
                            depth=1.0,
                            vertices=16,
                            location=(0, 0, 0),
                            rotation=(0, 0, 0),
                            enter_editmode=False
                        )
                        capsule_obj = bpy.context.active_object
                        capsule_obj.name = capsule_name
                        capsule_obj["tr7ae_type"] = "HCapsule"
                        capsule_obj["tr7ae_bone_index"] = i

                        radius = cap.get("radius", 1.0)
                        length = cap.get("length", 1.0)
                        capsule_obj.scale = (radius, radius, length)

                        capsule_obj.display_type = 'WIRE'
                        capsule_obj.show_in_front = True

                        pose_bone = armature_obj.pose.bones[bone_name]
                        bone_mat = pose_bone.matrix

                        local_pos = Vector((
                            cap.get("posx", 0.0),
                            cap.get("posy", 0.0),
                            cap.get("posz", 0.0)
                        ))
                        world_pos = bone_mat @ local_pos

                        capsule_obj.location = (
                            hinfo_obj.matrix_world.inverted() @ world_pos / 100
                        )

                        capsule_obj.rotation_mode = 'QUATERNION'
                        capsule_obj.rotation_quaternion = Quaternion((
                            cap.get("quatw", 1.0),
                            cap.get("quatx", 0.0),
                            cap.get("quaty", 0.0),
                            cap.get("quatz", 0.0),
                        ))

                        capsule_obj.parent = hinfo_obj
                        capsule_obj.matrix_parent_inverse.identity()

                        vg = capsule_obj.vertex_groups.new(name=bone_name)
                        bpy.context.view_layer.objects.active = capsule_obj
                        bpy.ops.object.mode_set(mode='EDIT')
                        bpy.ops.mesh.select_all(action='SELECT')
                        bpy.ops.object.mode_set(mode='OBJECT')
                        vg.add(range(len(capsule_obj.data.vertices)), 1.0, 'REPLACE')

                        arm_mod = capsule_obj.modifiers.new(name="Armature", type='ARMATURE')
                        arm_mod.object = armature_obj

        if do_import_cloth:
            try_import_cloth(filepath, armature_obj)

        drying_groups_by_vertex = {}

        if mface_list != 0:
            fhandle.seek(mface_list + section_info_size)
            mface_faces = []
            mface_samebits = []

            for _ in range(num_faces):
                face_bytes = fhandle.read(8)
                if len(face_bytes) < 8:
                    break
                v0, v1, v2, same = struct.unpack("<4H", face_bytes)
                mface_faces.append((v0, v1, v2))
                mface_samebits.append(same)

            for (v0, v1, v2), same in zip(mface_faces, mface_samebits):
                gid0 = (same >> 0) & 0b11111
                gid1 = (same >> 5) & 0b11111
                gid2 = (same >> 10) & 0b11111

                drying_groups_by_vertex[v0] = gid0
                drying_groups_by_vertex[v1] = gid1
                drying_groups_by_vertex[v2] = gid2

        for mesh_index, (chunk_verts, chunk_faces, chunk_uvs, chunk_normals, chunk_segments, draw_group, chunk_vert_map, texture_id, blend_value, unknown_1, unknown_2, single_sided, texture_wrap, unknown_3, unknown_4, flat_shading, sort_z, stencil_pass, stencil_func, alpha_ref) in enumerate(mesh_chunks):
            mesh = bpy.data.meshes.new(f"TR7AE_Mesh_{mesh_index}")
            mesh.from_pydata(chunk_verts, [], chunk_faces)
            mesh.update()

            if drying_groups_by_vertex:
                vcol = mesh.vertex_colors.new(name="MFace")

                for poly in mesh.polygons:
                    for loop_idx in poly.loop_indices:
                        vert_idx = mesh.loops[loop_idx].vertex_index

                        original_idx = next((k for k, v in chunk_vert_map.items() if v == vert_idx), None)
                        gid = drying_groups_by_vertex.get(original_idx, 0)

                        random.seed(gid)
                        color = (random.random(), random.random(), random.random(), 1.0)
                        vcol.data[loop_idx].color = color

            if vertex_colors:
                color_layer = mesh.vertex_colors.new(name="Color")
                color_data = color_layer.data
                for poly in mesh.polygons:
                    for loop_idx in poly.loop_indices:
                        vertex_idx = mesh.loops[loop_idx].vertex_index
                        original_idx = list(chunk_vert_map.keys())[list(chunk_vert_map.values()).index(vertex_idx)]
                        color_data[loop_idx].color = vertex_colors[original_idx]
            uses_envmap = any(orig_idx in envmapped_indices for orig_idx in chunk_vert_map.keys())
            uses_eyerefenvmap = any(orig_idx in eyerefenvmapped_indices for orig_idx in chunk_vert_map.keys())

            mesh_name = f"Mesh_{mesh_index}"
            mesh_obj = bpy.data.objects.new(mesh_name, mesh)
            mesh.tr7ae_is_envmapped = uses_envmap
            mesh.tr7ae_is_eyerefenvmapped = uses_eyerefenvmap
            if uses_envmap:
                mat = bpy.data.materials.new(name=f"Material_Env_{mesh_index}")
            elif uses_eyerefenvmap:
                mat = bpy.data.materials.new(name=f"Material_EyeRefEnv_{mesh_index}")
            else:
                mat = bpy.data.materials.new(name=f"Material_{mesh_index}")

            mat.use_transparent_shadow = False

            mat["tr7ae_is_envmapped"] = uses_envmap
            mat.tr7ae_is_envmapped = uses_envmap
            mat["tr7ae_is_eyerefenvmapped"] = uses_eyerefenvmap
            mat.tr7ae_is_eyerefenvmapped = uses_eyerefenvmap
            mat.use_nodes = True
            nodes = mat.node_tree.nodes
            links = mat.node_tree.links

            for node in nodes:
                nodes.remove(node)

            output = nodes.new(type='ShaderNodeOutputMaterial')
            bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
            bsdf.inputs['Specular IOR Level'].default_value = 0.0
            tex_image = nodes.new(type='ShaderNodeTexImage')

            output.location = (300, 0)
            bsdf.location = (0, 0)
            tex_image.location = (-300, 0)

            links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])

            import glob

            def find_texture_by_id(folder, texture_id):
                hex_id = f"{texture_id:x}"
                pattern = os.path.join(folder, f"*_{hex_id}.pcd")
                matches = glob.glob(pattern)
                return matches[0] if matches else None
            
            folder = os.path.dirname(filepath)
            pcd_path = find_texture_by_id(folder, texture_id)

            output.location = (300, 0)
            bsdf.location = (0, 0)
            tex_image.location = (-600, 0)

            links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])

            from pathlib import Path

            pcd_path = find_texture_by_id(folder, texture_id)

            image = None
            if do_import_textures:
                dds_key = str(pcd_path)
                if dds_key in converted_textures:
                    dds_path, dds_format = converted_textures[dds_key]
                else:
                    result = convert_pcd_to_dds(pcd_path, texture_dir)
                    if result:
                        dds_path, dds_format = result
                        converted_textures[dds_key] = (dds_path, dds_format)
                    else:
                        dds_path = None
                        dds_format = None

                if str(dds_path) in loaded_images:
                    image = loaded_images[str(dds_path)]
                elif dds_path.exists():
                    image = bpy.data.images.load(str(dds_path))
                    loaded_images[str(dds_path)] = image
                else:
                    image = None

                if image:
                    tex_image.image = image
                else:
                    print(f"[ERROR] DDS file not found or not written: {dds_path}")

            if vertex_colors:
                vc_node = nodes.new(type='ShaderNodeAttribute')
                vc_node.attribute_name = "Color"
                vc_node.location = (-800, 0)

                mult_node = nodes.new(type='ShaderNodeMixRGB')
                mult_node.blend_type = 'MULTIPLY'
                mult_node.inputs['Fac'].default_value = 1.0
                mult_node.location = (-200, 0)

                links.new(vc_node.outputs['Color'], mult_node.inputs['Color2'])
                links.new(tex_image.outputs['Color'], mult_node.inputs['Color1'])

                links.new(mult_node.outputs['Color'], bsdf.inputs['Base Color'])

            if uses_envmap or uses_eyerefenvmap:
                if blend_value == 2 or 8:
                    mat.use_nodes = True
                    mat.blend_method = 'BLEND'

                    nodes = mat.node_tree.nodes
                    links = mat.node_tree.links
                    nodes.clear()

                    output = nodes.new(type='ShaderNodeOutputMaterial')
                    output.location = (1000, 0)

                    principled = nodes.new(type='ShaderNodeBsdfPrincipled')
                    principled.location = (600, 0)
                    principled.inputs['Roughness'].default_value = 0.5

                    multiply_alpha = nodes.new(type='ShaderNodeMixRGB')
                    multiply_alpha.blend_type = 'MULTIPLY'
                    multiply_alpha.inputs['Fac'].default_value = 1.0
                    multiply_alpha.location = (200, -200)

                    multiply_boost = nodes.new(type='ShaderNodeMixRGB')
                    multiply_boost.blend_type = 'MULTIPLY'
                    multiply_boost.inputs['Fac'].default_value = 1.0
                    multiply_boost.inputs['Color2'].default_value = (5.0, 5.0, 5.0, 1.0)
                    multiply_boost.location = (200, 100)

                    tex_image = nodes.new(type='ShaderNodeTexImage')
                    tex_image.image = image
                    tex_image.interpolation = 'Linear'
                    tex_image.extension = 'EXTEND'
                    tex_image.projection = 'BOX'
                    tex_image.projection_blend = 1.0
                    tex_image.location = (-400, 100)

                    vc_node = nodes.new(type='ShaderNodeAttribute')
                    vc_node.attribute_name = "Color"
                    vc_node.location = (-400, -100)

                    mapping = nodes.new(type='ShaderNodeMapping')
                    mapping.vector_type = 'TEXTURE'
                    mapping.location = (-800, 0)
                    mapping.inputs['Location'].default_value = (-2.0, 0.0, -2.0)
                    mapping.inputs['Rotation'].default_value = (0.0, 0.0, 0.0)
                    mapping.inputs['Scale'].default_value = (4.0, 4.0, 4.0)

                    tex_coord = nodes.new(type='ShaderNodeTexCoord')
                    tex_coord.location = (-1000, 0)

                    links.new(tex_coord.outputs['Reflection'], mapping.inputs['Vector'])
                    links.new(mapping.outputs['Vector'], tex_image.inputs['Vector'])

                    links.new(tex_image.outputs['Color'], multiply_boost.inputs['Color1'])
                    links.new(multiply_boost.outputs['Color'], principled.inputs['Base Color'])

                    links.new(tex_image.outputs['Color'], multiply_alpha.inputs['Color1'])
                    links.new(vc_node.outputs['Alpha'], multiply_alpha.inputs['Color2'])
                    links.new(multiply_alpha.outputs['Color'], principled.inputs['Alpha'])

                    links.new(principled.outputs['BSDF'], output.inputs['Surface'])


                if blend_value == 1:
                    mat.use_nodes = True
                    mat.blend_method = 'BLEND'

                    nodes = mat.node_tree.nodes
                    links = mat.node_tree.links
                    nodes.clear()

                    output = nodes.new(type='ShaderNodeOutputMaterial')
                    output.location = (800, 0)

                    principled = nodes.new(type='ShaderNodeBsdfPrincipled')
                    principled.location = (400, 0)
                    principled.inputs['Roughness'].default_value = 0.5

                    tex_image = nodes.new(type='ShaderNodeTexImage')
                    tex_image.image = image
                    tex_image.interpolation = 'Linear'
                    tex_image.extension = 'EXTEND'
                    tex_image.projection = 'BOX'
                    tex_image.projection_blend = 1.0
                    tex_image.location = (-400, 100)

                    vc_node = nodes.new(type='ShaderNodeAttribute')
                    vc_node.attribute_name = "Color"
                    vc_node.location = (-400, -100)

                    mapping = nodes.new(type='ShaderNodeMapping')
                    mapping.vector_type = 'TEXTURE'
                    mapping.location = (-800, 0)
                    mapping.inputs['Location'].default_value = (-2.0, 0.0, -2.0)
                    mapping.inputs['Rotation'].default_value = (0.0, 0.0, 0.0)
                    mapping.inputs['Scale'].default_value = (4.0, 4.0, 4.0)

                    tex_coord = nodes.new(type='ShaderNodeTexCoord')
                    tex_coord.location = (-1000, 0)

                    links.new(tex_coord.outputs['Reflection'], mapping.inputs['Vector'])
                    links.new(mapping.outputs['Vector'], tex_image.inputs['Vector'])
                    links.new(tex_image.outputs['Color'], principled.inputs['Base Color'])
                    links.new(vc_node.outputs['Alpha'], principled.inputs['Alpha'])
                    links.new(principled.outputs['BSDF'], output.inputs['Surface'])

            mat.tr7ae_texture_id = texture_id
            mat.tr7ae_blend_value = blend_value
            mat.tr7ae_unknown_1 = unknown_1
            mat.tr7ae_unknown_2 = unknown_2
            mat.tr7ae_single_sided = bool(single_sided)
            if single_sided:
                mat.use_backface_culling = True
            mat.tr7ae_texture_wrap = texture_wrap
            mat.tr7ae_unknown_3 = unknown_3
            mat.tr7ae_unknown_4 = unknown_4
            mat.tr7ae_flat_shading = flat_shading
            mat.tr7ae_sort_z = sort_z
            mat.tr7ae_stencil_pass = stencil_pass
            mat.tr7ae_stencil_func = stencil_func
            mat.tr7ae_alpha_ref = alpha_ref

            if unknown_1 == 2:
                    mat.use_transparency_overlap = False

            mesh.materials.append(mat)
            context.collection.objects.link(mesh_obj)
            mesh_obj.parent = armature_obj
            mesh["tr7ae_draw_group"] = draw_group
            mesh.tr7ae_draw_group = draw_group
            if chunk_uvs:
                mesh.uv_layers.new(name="UVMap")
                uv_layer = mesh.uv_layers.active.data
                for poly in mesh.polygons:
                    for loop_index in range(poly.loop_start, poly.loop_start + poly.loop_total):
                        vertex_index = mesh.loops[loop_index].vertex_index
                        uv_layer[loop_index].uv = chunk_uvs[vertex_index]
            if chunk_normals:
                loop_normals = []
                for poly in mesh.polygons:
                    poly.use_smooth = True
                    for loop_index in range(poly.loop_start, poly.loop_start + poly.loop_total):
                        vertex_index = mesh.loops[loop_index].vertex_index
                        loop_normals.append(chunk_normals[vertex_index])
                mesh.normals_split_custom_set(loop_normals)
            modifier = mesh_obj.modifiers.new(name="Armature", type='ARMATURE')
            modifier.object = armature_obj
            for i in range(len(bones)):
                mesh_obj.vertex_groups.new(name=f"Bone_{i}")
            for i, seg in enumerate(chunk_segments):
                vg = mesh_obj.vertex_groups.get(f"Bone_{seg}")
                if vg:
                    vg.add([i], 1.0, 'REPLACE')
            for first_vertex, last_vertex, index, weight_index, weight in virtsegment_data:
                group_primary = mesh_obj.vertex_groups.get(f"Bone_{index}")
                group_secondary = mesh_obj.vertex_groups.get(f"Bone_{weight_index}")
                if group_primary and group_secondary:
                    for v in range(first_vertex, last_vertex + 1):
                        if v in chunk_vert_map:
                            local_index = chunk_vert_map[v]
                            group_secondary.add([local_index], weight, 'REPLACE')
                            group_primary.add([local_index], 1.0 - weight, 'ADD')
            # Remove unused vertex groups
            used_vertex_indices = {i for vg in mesh_obj.vertex_groups for i, w in enumerate(mesh_obj.data.vertices) if any(g.group == vg.index for g in w.groups)}
            for vg in list(mesh_obj.vertex_groups):
                if all(v.index not in used_vertex_indices or vg.index not in [g.group for g in v.groups] for v in mesh_obj.data.vertices):
                    mesh_obj.vertex_groups.remove(vg)

        if model_targets:
            create_model_target_visuals(model_targets, armature_obj)

        if markups:
            create_markup_visuals(markups, armature_obj)

class ImportTR7AEPS2(Operator, ImportHelper):
    """Import TR7AE .tr7aemesh model (PS2)"""
    bl_idname = "import_scene.tr7aemeshps2"
    bl_label = "Import TR7AE Mesh"
    bl_options = {'PRESET', 'UNDO'}

    filename_ext = ".tr7aemesh"
    filter_glob: StringProperty(default="*.tr7aemesh", options={'HIDDEN'})

    def execute(self, context):
        return self.read_tr7aemesh(self.filepath, context)

    def read_tr7aemesh(self, filepath, context):
        with open(filepath, "rb") as f:
            data = f.read()

        if not data.startswith(b"SECT"):
            self.report({'ERROR'}, "Invalid file format.")
            return {'CANCELLED'}

        def read_vec3(offset):
            x, y, z = struct.unpack_from("<fff", data, offset)
            return Vector((x, y, z))

        def read_vec3_short(offset):
            x, y, z = struct.unpack_from("<hhh", data, offset)
            return Vector((x, y, z))

        def mat_from_pivot(pivot):
            mat = Matrix.Identity(4)
            mat.translation = pivot
            return mat

        # === HEADER ===
        num_relocations = struct.unpack_from("<h", data, 0x0D)[0]
        SectionInfoSize = 24 + (num_relocations * 8)

        mesh_start = SectionInfoSize
        offset = mesh_start

        _ = struct.unpack_from("<I", data, offset)[0]
        bone_count, virtseg_count, bone_data_offset = struct.unpack_from("<3I", data, offset + 4)
        bone_data_offset += SectionInfoSize
        scaleX, scaleY, scaleZ = struct.unpack_from("<3f", data, offset + 16)

        offset = mesh_start + 0x4C

        # === BONES ===
        bone_struct_size = 0x20
        bones_raw = []
        bone_ranges = {}

        for i in range(bone_count):
            base = bone_data_offset + i * bone_struct_size
            pivot = Vector(struct.unpack_from("<fff", data, base))  # 12 bytes
            base += 12 + 4
            _ = struct.unpack_from("<I", data, base)[0]
            base += 4
            first_vtx, last_vtx, parent = struct.unpack_from("<3h", data, base)
            _ = struct.unpack_from("<H", data, base + 6)[0]  # skip 2 bytes
            _ = struct.unpack_from("<I", data, base + 8)[0]  # skip another 4

            mat = Matrix.Identity(4)
            mat.translation = pivot
            if i == 0:
                rot = Euler((1.5708, 0, 0)).to_matrix().to_4x4()  # 90 degrees X
                mat = rot @ mat

            bones_raw.append({
                "name": f"Bone_{i}",
                "index": i,
                "matrix": mat,
                "parent": parent,
                "pivot": pivot
            })
            bone_ranges[i] = (first_vtx, last_vtx)

        # === Apply hierarchy transforms like rapi.multiplyBones() ===
        world_matrices = [Matrix.Identity(4) for _ in range(len(bones_raw))]

        for i, bone in enumerate(bones_raw):
            mat = bone["matrix"]
            parent = bone["parent"]
            if parent >= 0:
                mat = world_matrices[parent] @ mat
            world_matrices[i] = mat

        # === MESH HEADER ===
        offset = mesh_start + 0x20
        num_vertices, vertex_list_offset = struct.unpack_from("<iI", data, offset)
        offset += 8
        num_faces, face_list_offset = struct.unpack_from("<iI", data, offset)

        # === VERTEX DATA ===
        offset = vertex_list_offset + SectionInfoSize
        raw_positions = []
        uvs = []

        for _ in range(num_vertices):
            x, y, z = struct.unpack_from("<hhh", data, offset)
            uv = data[offset + 6] / 128.0
            raw_positions.append(Vector((x, y, z)))
            uvs.append((uv, 0.0))
            offset += 8

        vert_weights = [None] * num_vertices
        transform_bone = [None] * num_vertices

        for bone_id, (first, last) in bone_ranges.items():
            if first == 0 and last == -1:
                continue
            for vtx in range(first, last + 1):
                if 0 <= vtx < num_vertices:
                    vert_weights[vtx] = [(bone_id, 1.0)]
                    transform_bone[vtx] = bone_id

        for i in range(virtseg_count):
            base = bone_data_offset + bone_count * bone_struct_size + i * bone_struct_size
            first_vtx, last_vtx, index, weight_index = struct.unpack_from("<4h", data, base + 0x14)
            weight = struct.unpack_from("<f", data, base + 0x1C)[0]
            for vtx in range(first_vtx, last_vtx + 1):
                if 0 <= vtx < num_vertices:
                    vert_weights[vtx] = [(index, 1.0 - weight), (weight_index, weight)]
                    transform_bone[vtx] = index

        for i in range(num_vertices):
            if vert_weights[i] is None:
                vert_weights[i] = [(0, 1.0)]
            if transform_bone[i] is None:
                transform_bone[i] = 0

        # === Apply bone transforms to vertices ===
        vertices = []
        for i, pos in enumerate(raw_positions):
            bone_id = transform_bone[i]
            mat = world_matrices[bone_id]
            transformed = mat @ pos.to_4d()
            vertices.append(Vector((transformed.x, transformed.y, transformed.z)))
        # Rotate all vertices -90 degrees around X axis
        rot_x_neg_90 = Matrix.Rotation(math.radians(-90), 4, 'X')
        vertices = [(rot_x_neg_90 @ v.to_4d()).to_3d() for v in vertices]
        scale_factor = 0.01
        vertices = [v * scale_factor for v in vertices]

        # === FACES ===
        offset = face_list_offset
        raw_indices = []

        while offset + 2 <= len(data):
            idx = struct.unpack_from("<H", data, offset)[0]
            raw_indices.append(idx)
            offset += 2

        strip_restart_values = {
            1057, 1058, 1089, 1090, 1091, 1121, 1122, 2081, 2082, 2113,
            2114, 2146, 2147, 3171, 2116, 2356, 2479, 2023, 2028, 4164,
            4130, 3138, 3170, 1819, 1765, 2180, 1833, 1154, 1828, 5219,
            1784, 2179, 1745, 2033, 4195, 1719, 1712, 1699, 4161, 4162,
            2178, 2041, 2059, 3172, 2065, 1769, 1823, 1092, 2148, 2115,
            1750, 3267, 1832, 1780, 2084, 1682, 1676, 2052, 2056, 2062,
            2079, 2085, 1624, 1628, 1637, 1635, 4197, 1645, 1655, 4227,
            1668, 4226
        }

        indices = []
        strip = []
        tri_count = 0

        for idx in raw_indices:
            if idx in strip_restart_values:
                strip = []
                tri_count = 0
                continue
            strip.append(idx)
            if len(strip) >= 3:
                i0, i1, i2 = strip[-3:]
                if tri_count % 2 == 0:
                    indices.append((i0, i1, i2))
                else:
                    indices.append((i1, i0, i2))
                tri_count += 1

        clean_faces = [f for f in indices if all(0 <= i < num_vertices for i in f)]

        # === CREATE ARMATURE ===
        armature_data = bpy.data.armatures.new("TR7AE_Armature")
        armature_obj = bpy.data.objects.new("TR7AE_Armature", armature_data)
        context.collection.objects.link(armature_obj)
        bpy.context.view_layer.objects.active = armature_obj
        bpy.ops.object.mode_set(mode='EDIT')

        edit_bones = armature_data.edit_bones
        bone_objects = {}

        # Accumulate world positions
        scale_factor = 0.01  # adjust if needed
        world_positions = [Vector((0, 0, 0)) for _ in range(bone_count)]

        for i, bone in enumerate(bones_raw):
            pivot = bone["pivot"] * scale_factor
            parent = bone["parent"]
            if parent < 0:
                world_positions[i] = pivot
            else:
                world_positions[i] = world_positions[parent] + pivot

        # Create edit bones using world positions
        for i, bone in enumerate(bones_raw):
            b = edit_bones.new(bone["name"])
            head = world_positions[i]
            tail = head + Vector((0, 0.05, 0))  # minimal visible length
            b.head = head
            b.tail = tail
            bone_objects[i] = b

        for i, bone in enumerate(bones_raw):
            parent_idx = bone["parent"]
            if parent_idx >= 0:
                bone_objects[i].parent = bone_objects[parent_idx]

        bpy.ops.object.mode_set(mode='OBJECT')

        # === CREATE MESH ===
        mesh = bpy.data.meshes.new("TR7AE_Mesh")
        mesh.from_pydata(vertices, [], clean_faces)
        mesh.update()

        obj = bpy.data.objects.new("TR7AE_Model", mesh)
        context.collection.objects.link(obj)

        # === UVs ===
        if not mesh.uv_layers:
            mesh.uv_layers.new(name="UVMap")
        uv_layer = mesh.uv_layers.active.data
        for poly in mesh.polygons:
            for loop_idx in poly.loop_indices:
                vidx = mesh.loops[loop_idx].vertex_index
                uv_layer[loop_idx].uv = uvs[vidx]

        # === SKINNING ===
        modifier = obj.modifiers.new(name="Armature", type='ARMATURE')
        modifier.object = armature_obj

        for i, weights in enumerate(vert_weights):
            for bone_idx, weight in weights:
                group_name = bones_raw[bone_idx]["name"]
                group = obj.vertex_groups.get(group_name)
                if not group:
                    group = obj.vertex_groups.new(name=group_name)
                group.add([i], weight, 'REPLACE')

        obj.parent = armature_obj
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        armature_obj.select_set(True)

        return {'FINISHED'}

class ImportPBRWC(Operator, ImportHelper):
    """Import Tomb Raider Legend/Anniversary .pbrwc model (PS3)"""
    bl_idname = "import_scene.pbrwc"
    bl_label = "Import PBRWC Model"
    bl_options = {'PRESET', 'UNDO'}

    filename_ext = ".pbrwc"
    filter_glob: StringProperty(default="*.pbrwc", options={'HIDDEN'})

    def execute(self, context):
        return self.read_pbrwc(self.filepath, context)

    def read_pbrwc(self, filepath, context):
        import struct
        from mathutils import Vector

        with open(filepath, "rb") as f:
            data = f.read()

        if len(data) < 24:
            self.report({'ERROR'}, "File too short.")
            return {'CANCELLED'}

        offset = 24
        try:
            vert_offset = struct.unpack(">I", data[offset + 4:offset + 8])[0] + offset + 4
            print(f"Vertex count offset: {vert_offset}")
            vert_count = struct.unpack(">I", data[vert_offset:vert_offset + 4])[0]
            vert_data_offset = struct.unpack(">I", data[vert_offset + 4:vert_offset + 8])[0] + offset + 8
            print(f"Vertex data offset: {vert_data_offset}")

            face_count = (vert_offset - offset - 8) // 2
            vertex_data_end = vert_data_offset + (vert_count * 0x10)

            vertices = []
            uvs = []
            faces = []

            # Vertex positions
            for i in range(vert_count):
                base = vert_data_offset + i * 0x10
                vx, vy, vz = struct.unpack(">hhh", data[base:base + 6])
                vertices.append(Vector((vx / 32767.0, vy / 32767.0, vz / 32767.0)))

            # NORMAL VECTOR CODE HERE. NORMAL VECTOR DATA MOST LIKELY STARTS AT OFFSET 8 IN THE STRIDE, STILL UNSURE.

            # UVs
            # THEY ARE STORED IN A DIFFERENT PLACE FOR SOME MODELS CHECK LATER!!!!
            extra_stride = 20
            uv_offset_in_block = 4

            for i in range(vert_count):
                base = vertex_data_end + i * extra_stride

                if base + uv_offset_in_block + 4 <= len(data):
                    u, v = struct.unpack(">hh", data[base + uv_offset_in_block : base + uv_offset_in_block + 4])
                    uvs.append(((u / 65535.0) * 16, (1.0 - (v / 65535.0)) * 16))
                else:
                    uvs.append((0.0, 0.0))

            # Faces
            face_data = data[offset + 8:offset + 8 + (face_count * 2)]
            face_indices = [struct.unpack(">H", face_data[i:i + 2])[0] for i in range(0, len(face_data), 2)]

            faces = []
            strip = []

            for idx in face_indices:
                if idx == 0xFFFF:
                    strip.clear()  # Restart strip
                    continue

                strip.append(idx)

                if len(strip) >= 3:
                    i = len(strip) - 3
                    a, b, c = strip[i], strip[i+1], strip[i+2]

                    # Skip degenerate triangles
                    if a == b or b == c or a == c:
                        continue

                    # Alternate winding within the strip
                    if i % 2 == 0:
                        faces.append((a, b, c))
                    else:
                        faces.append((b, a, c))


            self.create_mesh(vertices, faces, uvs, context)
            return {'FINISHED'}

        except Exception as e:
            self.report({'ERROR'}, f"Failed to parse .pbrwc: {str(e)}")
            return {'CANCELLED'}

    def create_mesh(self, vertices, faces, uvs, context):
        mesh = bpy.data.meshes.new("PBRWC_Model")
        obj = bpy.data.objects.new("PBRWC_Model", mesh)
        mesh.from_pydata(vertices, [], faces)
        mesh.update()

        # Assign UVs
        if not mesh.uv_layers:
            mesh.uv_layers.new(name="UVMap")
        uv_layer = mesh.uv_layers.active.data
        for poly in mesh.polygons:
            for loop_index in poly.loop_indices:
                vidx = mesh.loops[loop_index].vertex_index
                uv_layer[loop_index].uv = uvs[vidx]

        for poly in mesh.polygons:
            poly.use_smooth = True

        # Link to scene
        context.collection.objects.link(obj)
        context.view_layer.objects.active = obj
        obj.select_set(True)



def import_skeleton_from_tr7aemesh(largest_mesh_path):
    import bpy
    import struct
    from mathutils import Vector

    with open(largest_mesh_path, "rb") as f:
        if f.read(4) != b'SECT':
            print("Invalid .tr7aemesh file")
            return None

        f.seek(0xC)
        reloc_info = struct.unpack("<I", f.read(4))[0]
        num_relocs = (reloc_info >> 8) & 0xFFFFFF
        section_info_size = 0x18 + num_relocs * 8

        f.seek(section_info_size)
        version = struct.unpack("<I", f.read(4))[0]
        num_bones = struct.unpack("<I", f.read(4))[0]
        num_virtsegments = struct.unpack("<I", f.read(4))[0]
        bone_offset = struct.unpack("<I", f.read(4))[0]

        if bone_offset == 0 or num_bones == 0:
            print("No skeleton found in mesh.")
            return None

        # Seek to bone data
        f.seek(section_info_size + bone_offset)

        bones = []
        pivots = []
        parent_indices = []

        for i in range(num_bones):
            f.seek(section_info_size + bone_offset + i * 64 + 32)  # Pivot position
            pivot = struct.unpack("<3f", f.read(12))
            pivots.append(Vector(pivot))

            f.seek(section_info_size + bone_offset + i * 64 + 48)
            _ = f.read(4)  # flags
            f.read(4)      # vertex range
            parent_idx = struct.unpack("<i", f.read(4))[0]
            parent_indices.append(parent_idx)

        # Accumulate world positions
        scale_factor = 100.0
        world_positions = [Vector((0, 0, 0))] * num_bones
        for i in range(num_bones):
            pivot = pivots[i] * scale_factor
            parent = parent_indices[i]
            if parent < 0:
                world_positions[i] = pivot
            else:
                world_positions[i] = world_positions[parent] + pivot

        # Create Armature
        bpy.ops.object.add(type='ARMATURE', enter_editmode=True)
        arm_obj = bpy.context.active_object
        arm_obj.name = "Armature"
        arm_data = arm_obj.data
        arm_data.name = "Armature"

        bpy.ops.object.mode_set(mode='EDIT')
        edit_bones = arm_data.edit_bones

        bone_objects = {}

        for i in range(num_bones):
            name = f"Bone_{i}"
            head = world_positions[i]
            tail = head + Vector((0, 0.05, 0))  # Minimal visible length

            bone = edit_bones.new(name)
            bone.head = head
            bone.tail = tail
            bone_objects[i] = bone

        for i in range(num_bones):
            parent = parent_indices[i]
            if parent >= 0 and parent in bone_objects:
                bone_objects[i].parent = bone_objects[parent]

        bpy.ops.object.mode_set(mode='OBJECT')
        return arm_obj


def read_d3d_decl_type(f, dtype):
    if dtype == 0x0:   return struct.unpack("<f", f.read(4))          # FLOAT1
    elif dtype == 0x1: return struct.unpack("<2f", f.read(8))         # FLOAT2
    elif dtype == 0x2: return struct.unpack("<3f", f.read(12))        # FLOAT3
    elif dtype == 0x3: return struct.unpack("<4f", f.read(16))        # FLOAT4
    elif dtype == 0x4: return struct.unpack("<I", f.read(4))          # D3DCOLOR (packed BGRA)
    elif dtype == 0x5: return struct.unpack("<4B", f.read(4))         # UBYTE4
    elif dtype == 0x6: return struct.unpack("<2h", f.read(4))         # SHORT2
    elif dtype == 0x7: return struct.unpack("<4h", f.read(8))         # SHORT4
    elif dtype == 0x8: return struct.unpack("<4B", f.read(4))         # UBYTE4N
    elif dtype == 0x9: return tuple(i / 32767.0 for i in struct.unpack("<2h", f.read(4)))  # SHORT2N
    elif dtype == 0xA: return tuple(i / 32767.0 for i in struct.unpack("<4h", f.read(8)))  # SHORT4N
    elif dtype == 0xB: return tuple(i / 65535.0 for i in struct.unpack("<2H", f.read(4)))  # USHORT2N
    elif dtype == 0xC: return tuple(i / 65535.0 for i in struct.unpack("<4H", f.read(8)))  # USHORT4N
    elif dtype == 0xF: return struct.unpack("<2e", f.read(4))         # FLOAT16_2
    elif dtype == 0x10:return struct.unpack("<4e", f.read(8))         # FLOAT16_4
    else: return None  # Unhandled or UNUSED

usage_map = {
    0x0: "POSITION",
    0x1: "BLENDWEIGHT",
    0x2: "BLENDINDICES",
    0x3: "NORMAL",
    0x4: "PSIZE",
    0x5: "TEXCOORD",
    0x6: "TANGENT",
    0x7: "BINORMAL",
    0x8: "TESSFACTOR",
    0x9: "POSITIONT",
    0xA: "COLOR",
    0xB: "FOG",
    0xC: "DEPTH",
    0xD: "SAMPLE",
}

converted_textures = {}
loaded_images = {}

def load_texture_for_material(tex_id, model_dir, texture_dir, convert_pcd_to_dds, converted_textures, loaded_images):
    if tex_id in converted_textures:
        dds_path = converted_textures[tex_id]
    else:
        hex_id = f"{tex_id:x}"
        matches = list(model_dir.glob(f"*_{hex_id}.pcd"))
        if not matches:
            print(f"[Textures] .pcd not found for texture ID {hex_id}")
            return None

        pcd_path = matches[0]
        result = convert_pcd_to_dds(pcd_path, texture_dir)
        if not result:
            return None
        dds_path, _ = result
        converted_textures[tex_id] = dds_path

    if not dds_path or not dds_path.exists():
        return None

    if dds_path in loaded_images:
        return loaded_images[dds_path]

    try:
        image = bpy.data.images.load(str(dds_path), check_existing=True)
        loaded_images[dds_path] = image
        return image
    except Exception as e:
        print(f"[Textures] Failed to load {dds_path.name}: {e}")
        return None

class TR7AE_OT_ImportNextGenModel(bpy.types.Operator):
    bl_idname = "tr7ae.import_nextgen_model"
    bl_label = "Import TR7AE Next Gen Model"
    bl_description = "Import a Tomb Raider Legend/Anniversary Next Gen Model"
    bl_options = {'REGISTER', 'UNDO'}

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    filter_glob: bpy.props.StringProperty(
        default="*.tr7mesh",
        options={'HIDDEN'}
    )

    def execute(self, context):
        converted_textures.clear()
        loaded_images.clear()
        try:
            with open(self.filepath, "rb") as f:
                import pathlib
                model_dir = pathlib.Path(self.filepath).parent
                largest_mesh = max(model_dir.glob("*.tr7aemesh"), key=lambda p: p.stat().st_size, default=None)

                arm_obj = None
                if largest_mesh:
                    print(f"Importing skeleton from: {largest_mesh.name}")
                    arm_obj = import_skeleton_from_tr7aemesh(str(largest_mesh))
                else:
                    print("No .tr7aemesh file found to load skeleton.")

                if f.read(4) != b'SECT':
                    self.report({'ERROR'}, "Not a valid SECT file.")
                    return {'CANCELLED'}

                total_size = struct.unpack("<I", f.read(4))[0]
                f.seek(0xC)
                reloc_info = struct.unpack("<I", f.read(4))[0]
                num_relocs = (reloc_info >> 8) & 0xFFFFFF

                relocations = []
                f.seek(0x18)
                for _ in range(num_relocs):
                    section_index = struct.unpack("<H", f.read(2))[0] >> 3
                    _ = f.read(2)
                    offset = struct.unpack("<I", f.read(4))[0]
                    relocations.append((section_index, offset))

                model_extra_data_start = f.tell()

                offset_pc_model_data = struct.unpack("<I", f.read(4))[0]
                offset_pixmaps = struct.unpack("<I", f.read(4))[0]
                unk2 = struct.unpack("<I", f.read(4))[0]
                offset_special_material_flags = struct.unpack("<I", f.read(4))[0]

                offset_special_material_flags_pos = f.tell()

                # Read texture IDs (pixmaps)
                f.seek(model_extra_data_start + offset_pixmaps)
                texture_count = struct.unpack("<I", f.read(4))[0]
                texture_ids = list(struct.unpack(f"<{texture_count}I", f.read(texture_count * 4)))

                f.seek(offset_special_material_flags_pos)

                model_data_start = f.tell()

                magic_number = struct.unpack("<I", f.read(4))[0]
                flags = struct.unpack("<I", f.read(4))[0]
                total_data_size = struct.unpack("<I", f.read(4))[0]
                num_indices = struct.unpack("<I", f.read(4))[0]

                def read_vector(file):
                    return struct.unpack("<4f", file.read(16))

                bounding_sphere_center = read_vector(f)
                box_min = read_vector(f)
                box_max = read_vector(f)
                bounding_sphere_radius = struct.unpack("<f", f.read(4))[0]
                model_type = struct.unpack("<I", f.read(4))[0]
                sort_bias = struct.unpack("<f", f.read(4))[0]

                prim_groups_offset = struct.unpack("<I", f.read(4))[0]
                model_batch_offset = struct.unpack("<I", f.read(4))[0]
                bone_offset = struct.unpack("<I", f.read(4))[0]
                material_offset = struct.unpack("<I", f.read(4))[0]
                index_offset = struct.unpack("<I", f.read(4))[0]
                stencil_data_offset = struct.unpack("<I", f.read(4))[0]

                num_prim_groups, num_model_batches = struct.unpack("<2H", f.read(4))
                num_bones, num_materials = struct.unpack("<2H", f.read(4))
                num_pixmaps, highest_bend_index = struct.unpack("<2H", f.read(4))

                prim_groups = []
                f.seek(model_data_start + prim_groups_offset)
                for i in range(num_prim_groups):
                    base_index = struct.unpack("<I", f.read(4))[0]
                    num_primitives = struct.unpack("<I", f.read(4))[0]
                    num_vertices = struct.unpack("<I", f.read(4))[0]
                    vertex_shader_flags = struct.unpack("<H", f.read(2))[0]
                    f.read(2)
                    material_index = struct.unpack("<I", f.read(4))[0]

                    prim_groups.append({
                        "base_index": base_index,
                        "num_primitives": num_primitives,
                        "num_vertices": num_vertices,
                        "material_index": material_index,
                        "vertex_shader_flags": vertex_shader_flags,
                    })

                material_vertex_shader_flags = {}
                for group in prim_groups:
                    mat_idx = group["material_index"]
                    if mat_idx not in material_vertex_shader_flags:
                        material_vertex_shader_flags[mat_idx] = group["vertex_shader_flags"]


                materials = []
                f.seek(model_data_start + material_offset)
                for i in range(num_materials):
                    mat_id = struct.unpack("<i", f.read(4))[0]
                    f.read(20)
                    blend_mode = struct.unpack("<I", f.read(4))[0]
                    combiner_type = struct.unpack("<I", f.read(4))[0]
                    flags = struct.unpack("<I", f.read(4))[0]
                    opacity = struct.unpack("<f", f.read(4))[0]
                    poly_flags = struct.unpack("<I", f.read(4))[0]
                    uv_auto_scroll_speed = struct.unpack("<H", f.read(2))[0]
                    f.read(2)
                    sort_bias = struct.unpack("<f", f.read(4))[0]
                    detail_range_mul = struct.unpack("<f", f.read(4))[0]
                    detail_scale = struct.unpack("<f", f.read(4))[0]
                    parallax_scale = struct.unpack("<f", f.read(4))[0]
                    parallax_offset = struct.unpack("<f", f.read(4))[0]
                    specular_power = struct.unpack("<f", f.read(4))[0]
                    specular_shift_0 = struct.unpack("<f", f.read(4))[0]
                    specular_shift_1 = struct.unpack("<f", f.read(4))[0]
                    rim_light_color = struct.unpack("<4f", f.read(16))
                    rim_light_intensity = struct.unpack("<f", f.read(4))[0]
                    water_blend_bias = struct.unpack("<f", f.read(4))[0]
                    water_blend_exponent = struct.unpack("<f", f.read(4))[0]
                    water_deep_color = struct.unpack("<4f", f.read(16))
                    local_num_pixmaps = struct.unpack("<B", f.read(1))[0]
                    f.read(3)

                    layers = []
                    for layer_index in range(8):
                        color = struct.unpack("<4B", f.read(4))
                        tex_coord_source = struct.unpack("<i", f.read(4))[0]
                        modifier = struct.unpack("<i", f.read(4))[0]
                        param_id = struct.unpack("<i", f.read(4))[0]
                        constant = struct.unpack("<4f", f.read(16))
                        texture_index = struct.unpack("<H", f.read(2))[0]
                        num_textures = struct.unpack("<B", f.read(1))[0]
                        enabled = struct.unpack("<B", f.read(1))[0] != 0

                        if texture_index == 0xFFFF:
                            f.read((8 - layer_index - 1) * (36))
                            break

                        layers.append({
                            "color": color,
                            "tex_coord_source": tex_coord_source,
                            "modifier": modifier,
                            "param_id": param_id,
                            "constant": constant,
                            "texture_index": texture_index,
                            "num_textures": num_textures,
                            "enabled": enabled
                        })

                    f.read(40)
                    materials.append({
                        "mat_id": mat_id,
                        "blend_mode": blend_mode,
                        "combiner_type": combiner_type,
                        "flags": flags,
                        "opacity": opacity,
                        "poly_flags": poly_flags,
                        "uv_auto_scroll_speed": uv_auto_scroll_speed,
                        "sort_bias": sort_bias,
                        "detail_range_mul": detail_range_mul,
                        "detail_scale": detail_scale,
                        "parallax_scale": parallax_scale,
                        "parallax_offset": parallax_offset,
                        "specular_power": specular_power,
                        "rim_light_color": rim_light_color,
                        "rim_light_intensity": rim_light_intensity,
                        "specular_shift_0": specular_shift_0,
                        "specular_shift_1": specular_shift_1,
                        "water_blend_bias": water_blend_bias,
                        "water_blend_exponent": water_blend_exponent,
                        "water_deep_color": water_deep_color,
                        "layers": layers,
                    })

                material_cache = {}

                for i, mat_data in enumerate(materials):
                    mat = bpy.data.materials.new(name=f"TR7_Material_{i}")
                    mat.use_nodes = True
                    props = mat.nextgen_material_properties
                    props.vertex_shader_flags = material_vertex_shader_flags.get(i, 0)
                    props.mat_id = mat_data["mat_id"]
                    props.blend_mode = str(mat_data["blend_mode"])
                    props.combiner_type = str(mat_data["combiner_type"])
                    props.flags = mat_data["flags"]
                    props.opacity = mat_data["opacity"]
                    props.poly_flags = mat_data["poly_flags"]
                    props.uv_auto_scroll_speed = mat_data["uv_auto_scroll_speed"]
                    props.sort_bias = mat_data["sort_bias"]
                    props.detail_range_mul = mat_data["detail_range_mul"]
                    props.detail_scale = mat_data["detail_scale"]
                    props.parallax_scale = mat_data["parallax_scale"]
                    props.parallax_offset = mat_data["parallax_offset"]
                    props.specular_power = mat_data["specular_power"]
                    props.specular_shift_0 = mat_data["specular_shift_0"]
                    props.specular_shift_1 = mat_data["specular_shift_1"]
                    props.rim_light_color = mat_data["rim_light_color"]
                    props.rim_light_intensity = mat_data["rim_light_intensity"]
                    props.water_blend_bias = mat_data["water_blend_bias"]
                    props.water_blend_exponent = mat_data["water_blend_exponent"]
                    props.water_deep_color = mat_data["water_deep_color"]

                    props = mat.nextgen_material_properties
                    # (Your material property assignments here...)

                    nt = mat.node_tree
                    nodes = nt.nodes
                    links = nt.links

                    # Determine if this material has a specular (Layer 2)
                    has_specular = False
                    for layer_index, layer in enumerate(mat_data["layers"]):
                        if layer["enabled"] and layer_index == 2 and layer["texture_index"] != 0xFFFF:
                            has_specular = True
                            break

                    # Clear default nodes
                    for n in nodes:
                        nodes.remove(n)

                    # Output node
                    output = nodes.new("ShaderNodeOutputMaterial")
                    output.location = (800, 0)

                    # Transparency for alpha blending
                    transparent_node = nodes.new("ShaderNodeBsdfTransparent")
                    transparent_node.location = (-200, -500)

                    # Mix Shader for opacity blending
                    mix_shader = nodes.new("ShaderNodeMixShader")
                    mix_shader.location = (100, -200)

                    # Diffuse base
                    diffuse_node = nodes.new("ShaderNodeBsdfDiffuse")
                    diffuse_node.location = (-600, 300)

                    # Optional Glossy node
                    glossy_node = None
                    if has_specular:
                        glossy_node = nodes.new("ShaderNodeBsdfGlossy")
                        glossy_node.location = (-600, 0)
                        glossy_node.inputs["Roughness"].default_value = 0.3

                    # Add Shader to combine Diffuse + Glossy
                    add_shader = nodes.new("ShaderNodeAddShader")
                    add_shader.location = (-300, 200)

                    # Rim light setup
                    fresnel_node = nodes.new("ShaderNodeFresnel")
                    fresnel_node.location = (-600, -300)
                    fresnel_node.inputs["IOR"].default_value = 1

                    rim_color_node = nodes.new("ShaderNodeRGB")
                    rim_color_node.location = (-400, -300)
                    rim_color_node.outputs["Color"].default_value = (*mat_data["rim_light_color"][:3], 1.0)

                    rim_mult_node = nodes.new("ShaderNodeMixRGB")
                    rim_mult_node.location = (-200, -300)
                    rim_mult_node.blend_type = 'MULTIPLY'
                    rim_mult_node.inputs["Fac"].default_value = 1.0
                    links.new(rim_mult_node.inputs["Color1"], fresnel_node.outputs["Fac"])
                    links.new(rim_mult_node.inputs["Color2"], rim_color_node.outputs["Color"])

                    # Add rim to main shader
                    rim_add_shader = nodes.new("ShaderNodeAddShader")
                    rim_add_shader.location = (0, 200)

                    # Link rim output (diffuse + optional glossy + rim)
                    if has_specular and glossy_node:
                        links.new(add_shader.inputs[0], diffuse_node.outputs["BSDF"])
                        links.new(add_shader.inputs[1], glossy_node.outputs["BSDF"])
                        links.new(add_shader.outputs["Shader"], rim_add_shader.inputs[0])
                    else:
                        links.new(diffuse_node.outputs["BSDF"], rim_add_shader.inputs[0])
                    links.new(rim_mult_node.outputs["Color"], rim_add_shader.inputs[1])

                    # Connect transparency mix
                    links.new(mix_shader.inputs[1], rim_add_shader.outputs["Shader"])
                    links.new(mix_shader.inputs[2], transparent_node.outputs["BSDF"])
                    links.new(output.inputs["Surface"], mix_shader.outputs["Shader"])

                    # Blend mode logic
                    blend_mode = mat_data.get("blend_mode", 0)
                    gt_node = None

                    if blend_mode == 1:
                        gt_node = nodes.new("ShaderNodeMath")
                        gt_node.location = (-400, -600)
                        gt_node.operation = 'LESS_THAN'
                        gt_node.inputs[1].default_value = 0.5

                    elif blend_mode == 2:
                        mat.blend_method = 'BLEND'
                        links.new(mix_shader.inputs[1], rim_add_shader.outputs["Shader"])
                        links.new(mix_shader.inputs[2], transparent_node.outputs["BSDF"])
                        links.new(output.inputs["Surface"], mix_shader.outputs["Shader"])

                    else:
                        mix_shader.inputs["Fac"].default_value = 1.0 - mat_data["opacity"]

                    tex_x = -1800
                    tex_y = 300

                    for layer_index, layer in enumerate(mat_data["layers"]):
                        if not layer["enabled"]:
                            continue

                        texture_index = layer["texture_index"]
                        if texture_index == 0xFFFF or texture_index >= len(texture_ids):
                            continue

                        tex_id = texture_ids[texture_index]
                        image = load_texture_for_material(
                            tex_id, model_dir, texture_dir, convert_pcd_to_dds,
                            converted_textures, loaded_images
                        )
                        if not image:
                            continue

                        # Convert BGR to RGB color
                        b, g, r, a = layer["color"]
                        layer_color_floats = [r / 255.0, g / 255.0, b / 255.0]

                        tex_node = nodes.new("ShaderNodeTexImage")
                        tex_node.location = (tex_x, tex_y)
                        tex_node.image = image
                        if layer_index in (1, 2):  # Normal or Specular
                            tex_node.image.colorspace_settings.name = 'Non-Color'

                        mix_node = nodes.new("ShaderNodeMixRGB")
                        mix_node.location = (tex_x + 200, tex_y)
                        mix_node.blend_type = 'MULTIPLY'
                        mix_node.inputs["Fac"].default_value = 1.0
                        mix_node.inputs["Color2"].default_value = (*layer_color_floats, 1.0)
                        links.new(mix_node.inputs["Color1"], tex_node.outputs["Color"])

                        if layer_index == 0:
                            links.new(diffuse_node.inputs["Color"], mix_node.outputs["Color"])

                            if blend_mode == 1 and gt_node:
                                links.new(gt_node.inputs[0], tex_node.outputs["Alpha"])
                                links.new(mix_shader.inputs["Fac"], gt_node.outputs["Value"])
                            elif blend_mode == 2:
                                invert_alpha_node = nodes.new("ShaderNodeInvert")
                                invert_alpha_node.location = (tex_x + 400, tex_y)
                                links.new(invert_alpha_node.inputs["Color"], tex_node.outputs["Alpha"])
                                links.new(mix_shader.inputs["Fac"], invert_alpha_node.outputs["Color"])

                        elif has_specular and layer_index == 2:
                            invert = nodes.new("ShaderNodeInvert")
                            invert.location = (tex_x + 400, tex_y)
                            links.new(invert.inputs["Color"], mix_node.outputs["Color"])
                            links.new(glossy_node.inputs["Roughness"], invert.outputs["Color"])

                            divide_node = nodes.new("ShaderNodeMath")
                            divide_node.name = "SpecularPower_Divide"
                            divide_node.label = "Specular Power Divider"
                            divide_node.location = (tex_x + 600, tex_y)
                            divide_node.operation = 'DIVIDE'
                            divide_node.inputs[1].default_value = mat_data["specular_power"] if mat_data["specular_power"] != 0 else 1.0


                            links.new(divide_node.inputs[0], mix_node.outputs["Color"])
                            links.new(glossy_node.inputs["Color"], divide_node.outputs["Value"])

                        elif layer_index == 1:
                            separate = nodes.new("ShaderNodeSeparateColor")
                            separate.location = (tex_x + 400, tex_y)

                            invert = nodes.new("ShaderNodeInvert")
                            invert.location = (tex_x + 600, tex_y)

                            combine = nodes.new("ShaderNodeCombineColor")
                            combine.location = (tex_x + 800, tex_y)

                            normal_map = nodes.new("ShaderNodeNormalMap")
                            normal_map.location = (tex_x + 1000, tex_y)

                            links.new(separate.inputs["Color"], mix_node.outputs["Color"])
                            links.new(invert.inputs["Color"], separate.outputs["Green"])
                            links.new(combine.inputs["Red"], separate.outputs["Red"])
                            links.new(combine.inputs["Green"], invert.outputs["Color"])
                            links.new(combine.inputs["Blue"], separate.outputs["Blue"])
                            links.new(normal_map.inputs["Color"], combine.outputs["Color"])

                            links.new(diffuse_node.inputs["Normal"], normal_map.outputs["Normal"])
                            if has_specular and glossy_node:
                                links.new(glossy_node.inputs["Normal"], normal_map.outputs["Normal"])

                        tex_y -= 300

                    material_cache[i] = mat




                model_batches = []
                for batch_index in range(num_model_batches):
                    f.seek(model_data_start + model_batch_offset + batch_index * 0xAC)

                    flags = struct.unpack("<I", f.read(4))[0]
                    num_prim_groups = struct.unpack("<I", f.read(4))[0]
                    prim_groups_this_batch = prim_groups[:num_prim_groups]
                    prim_groups = prim_groups[num_prim_groups:]
                    skin_map_size = struct.unpack("<H", f.read(2))[0]
                    f.read(2)
                    skin_map_offset = struct.unpack("<I", f.read(4))[0]
                    vertex_data_offset = struct.unpack("<I", f.read(4))[0]
                    p_vertex_buffer = struct.unpack("<I", f.read(4))[0]

                    vertex_elements = []
                    seen_end = False

                    for _ in range(16):
                        stream, offset = struct.unpack("<HH", f.read(4))
                        dtype = struct.unpack("<B", f.read(1))[0]
                        method = struct.unpack("<B", f.read(1))[0]
                        usage = struct.unpack("<B", f.read(1))[0]
                        usage_index = struct.unpack("<B", f.read(1))[0]

                        if not seen_end:
                            if stream == 0xFF:
                                seen_end = True
                            else:
                                vertex_elements.append({
                                    "stream": stream,
                                    "offset": offset,
                                    "type": dtype,
                                    "method": method,
                                    "usage": usage,
                                    "usage_index": usage_index
                                })

                    vertex_format = struct.unpack("<I", f.read(4))[0]
                    vertex_stride = struct.unpack("<I", f.read(4))[0]
                    num_vertices = struct.unpack("<I", f.read(4))[0]
                    base_index = struct.unpack("<I", f.read(4))[0]
                    num_primitives = struct.unpack("<I", f.read(4))[0]

                    f.seek(model_data_start + skin_map_offset)
                    skin_map = struct.unpack(f"<{skin_map_size}I", f.read(skin_map_size * 4))

                    vertex_buffer_offset = model_data_start + vertex_data_offset
                    f.seek(vertex_buffer_offset)
                    vertex_data = f.read(num_vertices * vertex_stride)

                    vertex_list = []
                    for vtx_index in range(num_vertices):
                        vtx_data = vertex_data[vtx_index * vertex_stride : (vtx_index + 1) * vertex_stride]
                        vtx_stream = io.BytesIO(vtx_data)
                        vtx = {}
                        bone_weight = None
                        blend_indices = None

                        for elem in vertex_elements:
                            if elem["stream"] == 255 or elem["stream"] != 0:
                                continue

                            try:
                                vtx_stream.seek(elem["offset"])
                                val = read_d3d_decl_type(vtx_stream, elem["type"])
                                if val is not None:
                                    key = usage_map.get(elem["usage"], f"UNKNOWN_{elem['usage']}")
                                    if elem["usage_index"] > 0:
                                        key += f"_{elem['usage_index']}"

                                    if key == "BLENDWEIGHT":
                                        bone_weight = val[0] if isinstance(val, tuple) else val
                                    elif key == "BLENDINDICES":
                                        blend_indices = val
                                    else:
                                        vtx[key] = val
                            except Exception as e:
                                print(f"Error reading vertex element: {e}")

                        if blend_indices is not None:
                            b0, b1 = blend_indices
                            g0 = skin_map[b0] if b0 < len(skin_map) else 0
                            g1 = skin_map[b1] if b1 < len(skin_map) else 0

                            if b0 == b1 or bone_weight is None:
                                weights = [1.0]
                                indices = [g0]
                            else:
                                weights = [1.0 - bone_weight, bone_weight]
                                indices = [g0, g1]

                            vtx["BONE_WEIGHTS"] = weights
                            vtx["BONE_INDICES"] = indices

                        vertex_list.append(vtx)

                    model_batches.append({
                        "vertex_elements": vertex_elements,
                        "vertex_data": vertex_data,
                        "vertex_stride": vertex_stride,
                        "num_vertices": num_vertices,
                        "num_primitives": num_primitives,
                        "flags": flags,
                        "prim_groups": prim_groups_this_batch,
                        "base_index": base_index,
                        "decoded_vertices": vertex_list,
                        "skin_map_offset": skin_map_offset,
                        "skin_map_size": skin_map_size,
                        "skin_map": skin_map,
                    })

                f.seek(model_data_start + index_offset)
                index_data = struct.unpack(f"<{num_indices}H", f.read(num_indices * 2))

                def unpack_d3dcolor(packed: int):
                    r = (packed >> 0) & 0xFF
                    g = (packed >> 8) & 0xFF
                    b = (packed >> 16) & 0xFF
                    a = (packed >> 24) & 0xFF
                    return (r / 255.0, g / 255.0, b / 255.0, a / 255.0)

                for batch_index, batch in enumerate(model_batches):
                    for group_index, group in enumerate(batch["prim_groups"]):
                        base_index = group["base_index"]
                        num_primitives = group["num_primitives"]
                        start_face_index = base_index
                        end_face_index = base_index + num_primitives * 3
                        indices = index_data[start_face_index:end_face_index]

                        # Create mapping for index remapping
                        decoded_vertices = batch["decoded_vertices"]
                        used_indices = sorted(set(indices))
                        index_map = {global_idx: local_idx for local_idx, global_idx in enumerate(used_indices)}

                        # Build vertex list
                        verts = [Vector(decoded_vertices[i]["POSITION"][:3]) for i in used_indices]

                        # Build face list with corrected winding
                        faces = []
                        normals = []

                        double_sided_detected = False

                        for i in range(0, len(indices), 3):
                            i0, i1, i2 = indices[i:i+3]
                            try:
                                v0 = Vector(decoded_vertices[i0]["POSITION"])
                                v1 = Vector(decoded_vertices[i1]["POSITION"])
                                v2 = Vector(decoded_vertices[i2]["POSITION"])

                                n0 = Vector(decoded_vertices[i0]["NORMAL"]).normalized()
                                n1 = Vector(decoded_vertices[i1]["NORMAL"]).normalized()
                                n2 = Vector(decoded_vertices[i2]["NORMAL"]).normalized()

                                edge1 = v1 - v0
                                edge2 = v2 - v0
                                geometric_normal = edge1.cross(edge2).normalized()
                                average_normal = (n0 + n1 + n2).normalized()

                                is_flipped = geometric_normal.dot(average_normal) < 0

                                if is_flipped:
                                    double_sided_detected = True
                                    # Skip import of flipped faces
                                    flipped_face_count += 1
                                    continue

                                if is_flipped:
                                    face = (index_map[i2], index_map[i1], index_map[i0])
                                else:
                                    face = (index_map[i0], index_map[i1], index_map[i2])

                                faces.append(face)
                            except Exception as e:
                                continue

                        # Create mesh and object
                        mesh = bpy.data.meshes.new(name=f"TR7PrimGroup_{batch_index}_{group_index}")
                        mesh.from_pydata(verts, [], faces)
                        mesh.update()

                        obj = bpy.data.objects.new(mesh.name, mesh)
                        bpy.context.collection.objects.link(obj)

                        # Assign material
                        material_index = group["material_index"]
                        if material_index in material_cache:
                            mat = material_cache[material_index]
                            props = mat.nextgen_material_properties
                            props.double_sided = double_sided_detected
                            obj.data.materials.append(material_cache[material_index])
                            mat.use_backface_culling = not double_sided_detected

                        if arm_obj:
                            obj.parent = arm_obj
                            obj.parent_type = 'ARMATURE'
                            arm_obj.show_in_front = True
                            arm_obj.data.display_type = 'STICK'

                        # Set custom loop normals
                        loop_normals = []
                        normal_lookup = {i: Vector(decoded_vertices[i]["NORMAL"]).normalized() for i in used_indices}
                        for loop in mesh.loops:
                            vidx = loop.vertex_index
                            real_idx = used_indices[vidx]
                            loop_normals.append(normal_lookup.get(real_idx, Vector((0.0, 0.0, 1.0))))

                        for poly in mesh.polygons:
                            poly.use_smooth = True

                        mesh.normals_split_custom_set(loop_normals)


                        uv_channels = [key for key in decoded_vertices[0].keys() if key.startswith("TEXCOORD")]
                        for uv_name in uv_channels:
                            uv_layer = mesh.uv_layers.new(name=uv_name)
                            for poly in mesh.polygons:
                                for loop_index in range(poly.loop_start, poly.loop_start + poly.loop_total):
                                    vidx = mesh.loops[loop_index].vertex_index
                                    i = used_indices[vidx]
                                    uv = decoded_vertices[i].get(uv_name, (0.0, 0.0))
                                    uv_layer.data[loop_index].uv = (uv[0], 1.0 - uv[1])


                        color_channels = [key for key in decoded_vertices[0].keys() if key.startswith("COLOR")]
                        for color_name in color_channels:
                            color_layer = mesh.color_attributes.new(name=color_name, type='BYTE_COLOR', domain='CORNER')
                            for poly in mesh.polygons:
                                for loop_index in range(poly.loop_start, poly.loop_start + poly.loop_total):
                                    vidx = mesh.loops[loop_index].vertex_index
                                    i = used_indices[vidx]
                                    raw = decoded_vertices[i].get(color_name)
                                    if isinstance(raw, tuple):
                                        if len(raw) == 1 and isinstance(raw[0], int):
                                            color = unpack_d3dcolor(raw[0])
                                        elif len(raw) == 4:
                                            color = tuple(min(max(c, 0.0), 1.0) for c in raw)
                                        else:
                                            color = (1.0, 1.0, 1.0, 1.0)
                                    else:
                                        color = (1.0, 1.0, 1.0, 1.0)
                                    color_layer.data[loop_index].color = color

                        vertex_groups = {}
                        for i in used_indices:
                            v = decoded_vertices[i]
                            indices = v.get("BONE_INDICES", [])
                            for bone_index in indices:
                                group_name = f"Bone_{bone_index}"
                                if group_name not in vertex_groups:
                                    vertex_groups[group_name] = obj.vertex_groups.new(name=group_name)

                        for local_idx, i in enumerate(used_indices):
                            v = decoded_vertices[i]
                            weights = v.get("BONE_WEIGHTS", [])
                            indices = v.get("BONE_INDICES", [])
                            for bone_index, weight in zip(indices, weights):
                                group_name = f"Bone_{bone_index}"
                                group = vertex_groups.get(group_name)
                                if group and weight > 0.0:
                                    group.add([local_idx], weight, 'REPLACE')

                if arm_obj:
                    arm_obj.scale = (0.0001, 0.0001, 0.0001)
                    arm_obj.show_in_front = True
                    arm_obj.data.display_type = 'STICK'

                    for child in arm_obj.children:
                        if child.type == 'MESH':
                            child.scale = (100, 100, 100)

                    bpy.ops.object.select_all(action='DESELECT')
                    arm_obj.select_set(True)
                    bpy.context.view_layer.objects.active = arm_obj
                    arm_obj.scale = (0.01, 0.01, 0.01)
                    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
                    arm_obj.scale = (0.01, 0.01, 0.01)
                    arm_obj.select_set(False)

                    for child in arm_obj.children:
                        if child.type == 'MESH':
                            child.select_set(True)
                            bpy.context.view_layer.objects.active = child
                            bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
                            child.select_set(False)


                self.report({'INFO'}, f"Parsed {len(model_batches)} model batch(es) with decoded vertices.")
                return {'FINISHED'}

        except Exception as e:
            self.report({'ERROR'}, f"Failed to import: {e}")
            return {'CANCELLED'}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}
    
class TR7AE_OT_ExportNextGenModel(bpy.types.Operator):
    bl_idname = "tr7ae.export_nextgen_model"
    bl_label = "Export TR7AE Next Gen Model"
    bl_description = "Export a TR7AE Next Gen model"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        self.report({'INFO'}, "Export TR7AE Next Gen Model (not yet implemented)")
        return {'FINISHED'}
    
def decode_normal_ubyte4(data):
    return tuple((b / 255.0) * 2.0 - 1.0 for b in struct.unpack_from('<4B', data[:4]))

class ImportUnderworldGOLModel(bpy.types.Operator, ImportHelper):
    bl_idname = "tru.import_trugol_model"
    bl_label = "Import TRUGOL Model"
    bl_description = "Import a Tomb Raider Underworld/Guardian of Light Model"
    bl_options = {'REGISTER', 'UNDO'}

    filename_ext = ".obj"
    filter_glob: bpy.props.StringProperty(default="*.obj", options={'HIDDEN'})

    def execute(self, context):
        filepath = Path(self.filepath)
        try:
            # --- Step 1: Read Main File ---
            with open(filepath, 'rb') as f:
                post_header_int = self.read_post_header_id(f)

            # --- Step 2: Find File by Post-Header ID ---
            file_id_hex = f"{post_header_int:x}"
            parent_dir = filepath.parent
            file_1 = self.find_file_by_id(parent_dir, file_id_hex)
            if not file_1:
                raise FileNotFoundError(f"No matching file found for ID: {file_id_hex}")

            # --- Step 3: Get Mesh Meta ID ---
            with open(file_1, 'rb') as f:
                section_header_end, ref_offset = self.read_section_header_and_find_offset(f, 164)
                if ref_offset is None:
                    raise ValueError("No relocation with offset 164 found.")
                f.seek(section_header_end + ref_offset)
                mesh_meta_id = struct.unpack('<I', f.read(4))[0]

            mesh_meta_hex = f"{mesh_meta_id:x}"
            file_2 = self.find_file_by_id(parent_dir, mesh_meta_hex)
            if not file_2:
                raise FileNotFoundError(f"No matching file found for Mesh Meta ID: {mesh_meta_hex}")

            # --- Step 4: Get cdcRenderModel ID from Model Struct ---
            with open(file_2, 'rb') as f:
                section_header_end, _ = self.read_section_header_and_find_offset(f)
                f.seek(section_header_end)
                model_struct = struct.unpack('<3iI4f iI iI iI iI ff III iI iI III', f.read(112))
                cdc_render_model = model_struct[25]

            cdc_render_hex = f"{cdc_render_model:x}"
            file_3 = self.find_file_by_id(parent_dir, cdc_render_hex)
            if not file_3:
                raise FileNotFoundError(f"No matching file found for cdcRenderModel ID: {cdc_render_hex}")

            print(f"Found final mesh data file: {file_3}")

            # --- Step 5: Read Mesh Header ---
            with open(file_3, 'rb') as f:
                POSITION_COMPONENT_ID = 0xD2F7D823
                NORMAL_COMPONENT_ID = 0x36F5E414
                UV_COMPONENT_IDS = [0x8317902A, 0x8E54B6F3, 0x8A95AB44]
                COLOR_COMPONENT_IDS = [0x7E7DD623, 0x733EF0FA]

                section_header_end, _ = self.read_section_header_and_find_offset(f)
                f.seek(section_header_end + 16)  # Skip 16 bytes to Mesh Header

                # Read Mesh Header (130 bytes)
                mesh_header = struct.unpack('<4I 12f f I 36x 4I 3H', f.read(130))
                offsetMatIDs = mesh_header[2]
                numIndices = mesh_header[3]
                offsetMeshGroup = mesh_header[18]
                offsetMeshInfo = mesh_header[19]
                offsetFaceInfo = mesh_header[21]
                numMeshGroup = mesh_header[22]
                numMesh = mesh_header[23]
                numBones = mesh_header[24]

                # --- Step X: Read Material IDs ---
                f.seek(section_header_end + 16 + offsetMatIDs)
                f.read(4)  # Skip 4 bytes

                numMaterials = struct.unpack('<i', f.read(4))[0]
                material_ids = [struct.unpack('<i', f.read(4))[0] for _ in range(numMaterials)]
                print(f"Found {numMaterials} material IDs: {material_ids}")

                import glob

                # --- Step X: Find .matd Material Files ---
                materials_dir = os.path.dirname(file_3)  # Assuming file_3 is your model file

                matd_files = {}
                for mat_id in material_ids:
                    mat_id_hex = format(mat_id, 'x')
                    pattern = os.path.join(materials_dir, f"*_{mat_id_hex}.matd")
                    found_files = glob.glob(pattern)
                    if found_files:
                        matd_files[mat_id] = found_files[0]
                        print(f"Found .matd for material ID {mat_id} (hex: {mat_id_hex}): {found_files[0]}")
                    else:
                        print(f"Warning: No .matd file found for material ID {mat_id} (hex: {mat_id_hex})")

                material_lookup = {}
                for mat_id in material_ids:
                    mat = bpy.data.materials.new(name=f"Material_{mat_id}")
                    mat.use_nodes = True
                    material_lookup[mat_id] = mat

                for mat_id, matd_path in matd_files.items():
                    try:
                        with open(matd_path, 'rb') as matf:
                            mat_section_header_end, _ = self.read_section_header_and_find_offset(matf)
                            matf.seek(mat_section_header_end + 82)

                            texture_count = struct.unpack('<H', matf.read(2))[0]
                            print(f"Material {mat_id} has {texture_count} textures")

                            placeholder_offset = struct.unpack('<I', matf.read(4))[0]
                            reloc_relative_pos = matf.tell() - 4 - mat_section_header_end

                            matf.seek(0)
                            _, tex_ref_offset = self.read_section_header_and_find_offset(matf, target_offset=reloc_relative_pos)

                            if tex_ref_offset is None:
                                print(f"Warning: Could not find relocation for Material {mat_id}")
                                continue

                            true_tex_offset = mat_section_header_end + tex_ref_offset
                            matf.seek(true_tex_offset)

                            diffuse_assigned = False
                            normal_assigned = False

                            for tex_idx in range(texture_count):
                                try:
                                    tex_id = struct.unpack('<H', matf.read(2))[0]
                                    matf.read(6)
                                    texture_type = struct.unpack('<B', matf.read(1))[0]
                                    matf.read(4)
                                    texture_slot = struct.unpack('<B', matf.read(1))[0]
                                    texture_filter = struct.unpack('<H', matf.read(2))[0]

                                    print(f"  Texture {tex_idx}: ID={tex_id}, Type={texture_type}, Slot={texture_slot}, Filter={texture_filter}")

                                    tex_id_hex = format(tex_id, 'x')
                                    tex_pattern = os.path.join(materials_dir, f"*_{tex_id_hex}.pcd")
                                    found_files = glob.glob(tex_pattern)

                                    if not found_files:
                                        print(f"    Warning: No texture file found for ID {tex_id} (hex: {tex_id_hex})")
                                        continue

                                    tex_path = found_files[0]
                                    dds_path, _ = convert_pcd_to_dds(tex_path, texture_dir)

                                    if not dds_path:
                                        print(f"    Warning: DDS conversion failed for {tex_path}")
                                        continue

                                    mat = material_lookup.get(mat_id)
                                    if not mat:
                                        print(f"Warning: No Blender material found for ID {mat_id}")
                                        continue

                                    nodes = mat.node_tree.nodes
                                    links = mat.node_tree.links
                                    bsdf = next((n for n in nodes if n.type == 'BSDF_PRINCIPLED'), None)

                                    if not bsdf:
                                        print(f"Warning: No BSDF node found for material {mat.name}")
                                        continue

                                    # 🟡 Diffuse Texture (Type 1)
                                    if texture_type == 1 and not diffuse_assigned:
                                        tex_node = nodes.new(type='ShaderNodeTexImage')
                                        tex_node.image = bpy.data.images.load(str(dds_path))
                                        tex_node.label = f"Diffuse_{tex_id}"
                                        tex_node.location = (-300, 300)

                                        links.new(tex_node.outputs['Color'], bsdf.inputs['Base Color'])

                                        with open(dds_path, 'rb') as dds_file:
                                            dds_file.seek(84)
                                            fourcc = dds_file.read(4)
                                            is_dxt5 = (fourcc == b'DXT5')
                                            is_argb = False
                                            if fourcc == b'\x00\x00\x00\x00':
                                                dds_file.seek(88)
                                                rgb_bit_count = struct.unpack('<I', dds_file.read(4))[0]
                                                r_mask = struct.unpack('<I', dds_file.read(4))[0]
                                                g_mask = struct.unpack('<I', dds_file.read(4))[0]
                                                b_mask = struct.unpack('<I', dds_file.read(4))[0]
                                                a_mask = struct.unpack('<I', dds_file.read(4))[0]
                                                is_argb = (rgb_bit_count == 32 and a_mask != 0)

                                        if is_dxt5 or is_argb:
                                            links.new(tex_node.outputs['Alpha'], bsdf.inputs['Alpha'])
                                            mat.blend_method = 'BLEND'
                                            mat.shadow_method = 'HASHED'
                                            mat.use_screen_refraction = False
                                            mat.use_backface_culling = False
                                            print(f"🌿 Alpha channel connected for {dds_path.name}")
                                        else:
                                            print(f"🧱 Skipped Alpha for non-alpha DDS: {dds_path.name}")

                                        diffuse_assigned = True

                                    # 🔵 Normal Map Texture (Type 3)
                                    elif texture_type == 3 and not normal_assigned:
                                        tex_node = nodes.new(type='ShaderNodeTexImage')
                                        tex_node.image = bpy.data.images.load(str(dds_path))
                                        tex_node.label = f"Normal_{tex_id}"
                                        tex_node.location = (-600, 100)
                                        tex_node.image.colorspace_settings.name = 'Non-Color'

                                        separate_node = nodes.new(type='ShaderNodeSeparateColor')
                                        separate_node.location = (-400, 100)

                                        invert_node = nodes.new(type='ShaderNodeInvert')
                                        invert_node.location = (-200, 50)
                                        invert_node.inputs['Fac'].default_value = 1.0

                                        combine_node = nodes.new(type='ShaderNodeCombineColor')
                                        combine_node.location = (0, 100)

                                        normal_map_node = nodes.new(type='ShaderNodeNormalMap')
                                        normal_map_node.label = f"NormalMap_{tex_id}"
                                        normal_map_node.location = (200, 100)

                                        links.new(tex_node.outputs['Color'], separate_node.inputs['Color'])
                                        links.new(separate_node.outputs['Red'], combine_node.inputs['Red'])
                                        links.new(separate_node.outputs['Blue'], invert_node.inputs['Color'])
                                        links.new(invert_node.outputs['Color'], combine_node.inputs['Blue'])
                                        links.new(tex_node.outputs['Alpha'], combine_node.inputs['Green'])

                                        links.new(combine_node.outputs['Color'], normal_map_node.inputs['Color'])
                                        links.new(normal_map_node.outputs['Normal'], bsdf.inputs['Normal'])

                                        print(f"🔵 Normal map connected for {dds_path.name}")
                                        normal_assigned = True

                                except Exception as e:
                                    print(f"  Error reading texture {tex_idx} in {matd_path}: {e}")
                                    break

                    except Exception as e:
                        print(f"Error reading {matd_path}: {e}")

                # --- Step X: Read MeshGroups ---
                f.seek(section_header_end + 16 + offsetMeshGroup)
                mesh_groups = []

                for i in range(numMeshGroup):
                    x, y, z, w = struct.unpack('<4f', f.read(16))
                    baseIndex, numFaces, numVertices, flags = struct.unpack('<4I', f.read(16))
                    drawGroupId, order = struct.unpack('<ii', f.read(8))
                    materialIndex = struct.unpack('<I', f.read(4))[0]
                    materialID = material_ids[materialIndex] if materialIndex < len(material_ids) else -1
                    f.read(20)  # Skip 20 bytes

                    mesh_groups.append({
                        'index': i,
                        'baseIndex': baseIndex,
                        'numFaces': numFaces,
                        'numVertices': numVertices,
                        'drawGroupId': drawGroupId,
                        'order': order,
                        'materialIndex': materialIndex,
                        'materialID': materialID,  # BOH
                        'flags': flags
                    })

                print(f"Loaded {len(mesh_groups)} MeshGroups")

                # --- Step 5: Find external relocation (bones) ---
                with open(file_2, 'rb') as f2:
                    section_header_end_2 = f2.tell()
                    magic, size = struct.unpack('<II', f2.read(8))
                    type_byte, skip_byte = struct.unpack('<BB', f2.read(2))
                    version_id = struct.unpack('<H', f2.read(2))[0]
                    packed_data = struct.unpack('<I', f2.read(4))[0]
                    relocation_table_size = (packed_data >> 8) & 0xFFFFFF
                    f2.read(8)  # id + specMask

                    relocation_counts = struct.unpack('<5I', f2.read(20))
                    num_external = relocation_counts[1]

                    # Skip internal relocations
                    for _ in range(relocation_counts[0]):
                        f2.read(8)

                    external_ref_offset = None
                    for i in range(num_external):
                        raw = struct.unpack('<Q', f2.read(8))[0]
                        sectionIndex = raw & 0x3FFF
                        offset = (raw >> 14) & 0xFFFFFF
                        referencedOffset = (raw >> 38) & 0x3FFFFFF

                        if external_ref_offset is None:
                            external_ref_offset = referencedOffset

                    # Skip remaining relocation types if needed
                    for count, size in zip(relocation_counts[2:], [4, 4, 4]):
                        for _ in range(count):
                            f2.read(size)

                # --- Step 6: Read bone count and skeleton data ---
                if external_ref_offset is not None:
                    with open(file_3, 'rb') as f3:
                        f3_section_header_end, _ = self.read_section_header_and_find_offset(f3)
                        final_seek = f3_section_header_end + external_ref_offset
                        f3.seek(final_seek)
                        
                        bone_count_pos = f3.tell()
                        bone_count = struct.unpack('<I', f3.read(4))[0]
                        pointer_after_bone_count = f3.tell()

                        adjusted_bone_ptr_pos = pointer_after_bone_count - section_header_end

                        f3.read(12)

                        f3.seek(0)
                        magic, size = struct.unpack('<II', f3.read(8))
                        type_byte, skip_byte = struct.unpack('<BB', f3.read(2))
                        version_id = struct.unpack('<H', f3.read(2))[0]
                        packed_data = struct.unpack('<I', f3.read(4))[0]
                        relocation_table_size = (packed_data >> 8) & 0xFFFFFF
                        f3.read(8)

                        if relocation_table_size > 0:
                            num_relocations = struct.unpack('<5I', f3.read(20))
                            found_referenced_offset = None

                            for i in range(num_relocations[0]):
                                reloc_offset, reloc_ref_offset = struct.unpack('<II', f3.read(8))

                                if reloc_offset == adjusted_bone_ptr_pos:
                                    found_referenced_offset = reloc_ref_offset
                                    print(f"Found referenced offset for skeleton: {found_referenced_offset}")
                                    break

                            if found_referenced_offset is None:
                                print("Warning: Could not locate skeleton data!")

                        if found_referenced_offset is not None:
                            skeleton_data_pos = section_header_end + found_referenced_offset
                            f3.seek(skeleton_data_pos)

                        bones = []
                        for i in range(bone_count):
                            f3.read(0x20)

                            boneX, boneZ, boneY = struct.unpack('<3f', f3.read(12))
                            f3.read(8)
                            fakeFirstVertex, fakeLastVertex = struct.unpack('<2h', f3.read(4))
                            boneParentID = struct.unpack('<i', f3.read(4))[0]
                            f3.read(4)

                            bones.append({
                                'index': i,
                                'name': f"Bone_{i}",
                                'position': (boneX, boneZ, boneY),
                                'parent': boneParentID if boneParentID >= 0 else None
                            })

                        bpy.ops.object.add(type='ARMATURE', enter_editmode=True)
                        armature_obj = bpy.context.active_object
                        armature_obj.name = "Armature"
                        armature_obj.show_in_front = True
                        armature_obj.data.display_type = 'STICK'
                        armature_obj.scale = (0.01, 0.01, 0.01)
                        armature = armature_obj.data
                        bone_lookup = {}

                        for bone in bones:
                            b = armature.edit_bones.new(bone['name'])
                            b.head = Vector((0.0, 0.0, 0.0))
                            b.tail = Vector((0.0, 0.1, 0.0))
                            bone_lookup[bone['index']] = b

                        for bone in bones:
                            this_bone = bone_lookup[bone['index']]
                            pos = Vector(bone['position'])

                            parent_index = bone['parent']
                            if parent_index is not None and parent_index in bone_lookup:
                                parent_bone = bone_lookup[parent_index]
                                parent_head = parent_bone.head
                                pos += parent_head
                                this_bone.parent = parent_bone

                            this_bone.head = pos
                            this_bone.tail = pos + Vector((0.0, 0.1, 0.0))

                        bpy.ops.object.mode_set(mode='OBJECT')


                        print("Skeleton imported successfully.")
                else:
                    print("No external relocation found for skeleton.")

                # Step 1: Read Mesh Info Blocks
                f.seek(section_header_end + 16 +offsetMeshInfo)
                mesh_infos = []
                for i in range(numMesh):
                    f.read(8)
                    mesh_group_count, skinMapSize, skinMapOffset, vertexOffset = struct.unpack('<4i', f.read(16))
                    f.read(12)
                    vertexComponentsOffset, numVertices, meshBaseIndex, numFaces = struct.unpack('<4i', f.read(16))

                    mesh_infos.append({
                        'mesh_group_count': int(mesh_group_count),
                        'vertexOffset': section_header_end + 16 + vertexOffset,
                        'vertexComponentsOffset': section_header_end + 16 + vertexComponentsOffset,
                        'numVertices': int(numVertices),
                        'numFaces': int(numFaces),
                        'skinMapOffset': section_header_end + 16 + skinMapOffset,
                        'skinMapSize': int(skinMapSize),
                        'meshBaseIndex': int(meshBaseIndex),
                    })

                # Step 2: Read Vertex Components for Each Mesh
                for mesh in mesh_infos:
                    f.seek(mesh['vertexComponentsOffset'])
                    f.read(8)
                    numVertexComponents = struct.unpack('<H', f.read(2))[0]
                    mesh_vert_stride = struct.unpack('<B', f.read(1))[0]
                    f.read(5)

                    entries = []
                    for i in range(numVertexComponents):
                        vertexComponent, entryValue, entryType, entryNull = struct.unpack('<I H B B', f.read(8))
                        entries.append({
                            'component': vertexComponent,
                            'entryValue': entryValue,
                            'entryType': entryType
                        })

                    mesh['entries'] = entries
                    mesh['stride'] = mesh_vert_stride

                # Step 2: Read Skin Maps for Each Mesh
                for mesh in mesh_infos:
                    # Read the skin map for this mesh
                    skin_map = []
                    if mesh['skinMapSize'] > 0:
                        f.seek(mesh['skinMapOffset'])
                        for _ in range(mesh['skinMapSize']):
                            bone_index = struct.unpack('<I', f.read(4))[0]
                            skin_map.append(bone_index)
                    mesh['skin_map'] = skin_map

                # Step 3: Read Vertex Data for Each Mesh
                for mesh_index, mesh in enumerate(mesh_infos):
                    stride = mesh['stride']
                    entries = mesh['entries']
                    skin_map = mesh.get('skin_map', [])
                    f.seek(mesh['vertexOffset'])
                    vertex_data = []

                    for v in range(mesh['numVertices']):
                        raw = f.read(stride)
                        vertex = {}

                        skin_indices_raw = None
                        skin_weights_raw = None

                        for entry in entries:
                            comp = entry['component']
                            offset = entry['entryValue']
                            typ = entry['entryType']
                            try:
                                data = raw[offset:offset + 12]
                                if comp == 0xD2F7D823:
                                    vertex[comp] = struct.unpack_from('<3f', data)
                                elif comp == 0x36F5E414:
                                    x_raw, y_raw, z_raw, w_raw = struct.unpack_from('<4B', raw, offset)
                                    nx = (x_raw / 255.0) * 2.0 - 1.0
                                    ny = (y_raw / 255.0) * 2.0 - 1.0
                                    nz = (z_raw / 255.0) * 2.0 - 1.0
                                    length = math.sqrt(nx * nx + ny * ny + nz * nz)
                                    if length != 0:
                                        nx /= length
                                        ny /= length
                                        nz /= length
                                    vertex[comp] = (nz, ny, nx)
                                elif comp == 1364646099:  # Skin Indices
                                    skin_indices_raw = struct.unpack_from('<4B', data[:4])
                                elif comp == 1223070144:  # Skin Weights
                                    skin_weights_raw = struct.unpack_from('<4B', data[:4])
                                elif typ == 6:
                                    vertex[comp] = tuple((b / 255.0) * 2.0 - 1.0 for b in struct.unpack_from('<4B', data[:4]))[:3]
                                elif typ == 7:
                                    vertex[comp] = struct.unpack_from('<4B', data[:4])
                                elif comp in UV_COMPONENT_IDS:
                                    u_raw, v_raw = struct.unpack_from('<2h', data[:4])
                                    u = u_raw / 2048.0
                                    v = v_raw / 2048.0
                                    vertex[comp] = (u, 1.0 - v)
                                elif comp in COLOR_COMPONENT_IDS:
                                    raw_data = struct.unpack_from('<4B', data[:4])
                                    vertex[comp] = tuple([c / 255.0 for c in raw_data])
                                else:
                                    vertex[comp] = None
                            except Exception as e:
                                print(f"  ERROR: Could not unpack component 0x{comp:X} at offset {offset}: {e}")
                                vertex[comp] = None

                        if skin_indices_raw and skin_weights_raw:
                            mapped_indices = [skin_map[idx] if idx < len(skin_map) else -1 for idx in skin_indices_raw]

                            weight_index_pairs = list(zip(skin_weights_raw, mapped_indices))

                            weight_index_pairs.sort()

                            bone_indices = [idx for _, idx in weight_index_pairs if idx >= 0]
                            bone_weights = [w / 255.0 for w, idx in weight_index_pairs if idx >= 0]

                            vertex['bone_indices'] = bone_indices
                            vertex['bone_weights'] = bone_weights

                        elif skin_indices_raw and not skin_weights_raw:
                            mapped_index = skin_map[skin_indices_raw[0]] if skin_indices_raw[0] < len(skin_map) else -1
                            if mapped_index >= 0:
                                vertex['bone_indices'] = [mapped_index]
                                vertex['bone_weights'] = [1.0]
                            else:
                                vertex['bone_indices'] = []
                                vertex['bone_weights'] = []

                        vertex_data.append(vertex)

                    mesh['vertex_data'] = vertex_data

                # --- Step 4: Read Skin Maps and Apply Weights ---
                for mesh_index, mesh in enumerate(mesh_infos):
                    skin_map_offset = mesh.get('skinMapOffset')
                    skin_map_size = mesh.get('skinMapSize')

                    # Read the skin map for this mesh
                    f.seek(skin_map_offset)
                    skin_map = [struct.unpack('<I', f.read(4))[0] for _ in range(skin_map_size)]
                    mesh['skin_map'] = skin_map

                    # Find vertex components for bone indices and weights
                    bone_indices_comp = next((e for e in mesh['entries'] if e['component'] == 1364646099), None)
                    bone_weights_comp = next((e for e in mesh['entries'] if e['component'] == 1223070144), None)

                    if not bone_indices_comp or not bone_weights_comp:
                        print(f"Mesh {mesh_index}: No skin indices or weights component found, skipping skin data.")
                        continue

                    # Apply skin indices and weights to vertex data
                    for v in mesh['vertex_data']:
                        indices_raw = v.get(bone_indices_comp['component'])
                        weights_raw = v.get(bone_weights_comp['component'])

                        if indices_raw and weights_raw:
                            # Indices: map skinMap index to real bone index
                            mapped_indices = [skin_map[i] if i < len(skin_map) else -1 for i in indices_raw]

                            # Weights: sort by value and normalize
                            weight_values = list(weights_raw)
                            weight_pairs = sorted(zip(weight_values, mapped_indices), reverse=True)  # Sort by weight descending

                            bone_indices = []
                            bone_weights = []
                            for w, i in weight_pairs:
                                if i >= 0:
                                    bone_indices.append(i)
                                    bone_weights.append(w / 255.0)

                            # Normalize weights if they don't sum to 1
                            total_weight = sum(bone_weights)
                            if total_weight > 0:
                                bone_weights = [w / total_weight for w in bone_weights]

                            v['bone_indices'] = bone_indices
                            v['bone_weights'] = bone_weights

                print(f"\nAll vertex data loaded successfully!")

                f.seek(section_header_end + 16 + offsetFaceInfo)
                print(f"\n--- Reading Face Data ---")

                # --- Step: Read All Faces Once ---
                f.seek(section_header_end + 16 + offsetFaceInfo)
                all_faces = []
                for _ in range(numIndices // 3):
                    tri_indices = struct.unpack('<3H', f.read(6))
                    all_faces.append(tri_indices)

                # --- Group mesh_groups under mesh_infos ---
                mesh_group_pointer = 0
                for mesh in mesh_infos:
                    count = mesh['mesh_group_count']
                    mesh['mesh_groups'] = mesh_groups[mesh_group_pointer : mesh_group_pointer + count]
                    mesh_group_pointer += count

                # --- Read global index buffer once ---
                f.seek(section_header_end + 16 + offsetFaceInfo)
                global_indices = [struct.unpack('<H', f.read(2))[0] for _ in range(numIndices)]

                # --- Create Blender Meshes per MeshGroup ---
                for mesh_index, mesh in enumerate(mesh_infos):
                    print(f"\n=== Building MeshGroups for MeshInfo[{mesh_index}] ===")
                    vertex_data = mesh['vertex_data']

                    for group in mesh['mesh_groups']:
                        base = group['baseIndex']
                        numFaces = group['numFaces']
                        start = base
                        end = start + numFaces * 3

                        group_indices = global_indices[start:end]
                        group_faces = [tuple(group_indices[i:i+3]) for i in range(0, len(group_indices), 3)]

                        if not group_faces:
                            print(f"  Skipping empty MeshGroup[{group['index']}]")
                            continue

                        used_vertex_indices = set(idx for tri in group_faces for idx in tri)
                        index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted(used_vertex_indices))}

                        positions = []
                        normals = []
                        uvs_data = {uv_id: [] for uv_id in UV_COMPONENT_IDS}
                        color_data = {col_id: [] for col_id in COLOR_COMPONENT_IDS}
                        bone_weights = []
                        bone_indices = []

                        for old_idx in sorted(used_vertex_indices):
                            v = vertex_data[old_idx] if old_idx < len(vertex_data) else {}

                            pos = v.get(POSITION_COMPONENT_ID, (0.0, 0.0, 0.0))
                            normal = v.get(NORMAL_COMPONENT_ID, (0.0, 0.0, 1.0))
                            positions.append(pos)
                            normals.append(normal)

                            for uv_id in UV_COMPONENT_IDS:
                                uvs_data[uv_id].append(v.get(uv_id, (0.0, 0.0)))

                            for col_id in COLOR_COMPONENT_IDS:
                                color = v.get(col_id, (1.0, 1.0, 1.0, 1.0))
                                color_data[col_id].append(color if isinstance(color, (list, tuple)) else (1.0, 1.0, 1.0, 1.0))

                            bone_indices.append(v.get('bone_indices', []))
                            bone_weights.append(v.get('bone_weights', []))

                        blender_faces = []
                        flipped_face_count = 0
                        total_faces = 0

                        for tri in group_faces:
                            i0, i1, i2 = tri
                            try:
                                v0 = Vector(vertex_data[i0].get(POSITION_COMPONENT_ID, (0, 0, 0)))
                                v1 = Vector(vertex_data[i1].get(POSITION_COMPONENT_ID, (0, 0, 0)))
                                v2 = Vector(vertex_data[i2].get(POSITION_COMPONENT_ID, (0, 0, 0)))

                                n0 = Vector(vertex_data[i0].get(NORMAL_COMPONENT_ID, (0, 0, 1))).normalized()
                                n1 = Vector(vertex_data[i1].get(NORMAL_COMPONENT_ID, (0, 0, 1))).normalized()
                                n2 = Vector(vertex_data[i2].get(NORMAL_COMPONENT_ID, (0, 0, 1))).normalized()

                                edge1 = v1 - v0
                                edge2 = v2 - v0
                                geometric_normal = edge1.cross(edge2).normalized()
                                average_normal = (n0 + n1 + n2).normalized()

                                is_flipped = geometric_normal.dot(average_normal) < 0

                                if is_flipped:
                                    flipped_face_count += 1
                                    continue

                                # Map indices to local Blender vertex indices
                                blender_faces.append([
                                    index_map[i0],
                                    index_map[i1],
                                    index_map[i2],
                                ])
                                total_faces += 1

                            except Exception as e:
                                print(f"Error processing face {tri}: {e}")
                                continue

                        # --- Create Blender Mesh ---
                        mesh_name = f"TRU_Mesh{mesh_index}_Group{group['index']}"
                        blender_mesh = bpy.data.meshes.new(mesh_name)
                        blender_object = bpy.data.objects.new(mesh_name, blender_mesh)
                        bpy.context.collection.objects.link(blender_object)
                        blender_object.parent = armature_obj
                        blender_object.parent_type = 'ARMATURE'

                        # --- Assign Material ---
                        material = material_lookup.get(group['materialID'])
                        if material:
                            if len(blender_mesh.materials) == 0:
                                blender_mesh.materials.append(material)
                            else:
                                blender_mesh.materials[0] = material
                            print(f"Assigned {material.name} to {mesh_name}")
                        else:
                            print(f"Warning: No material found for {mesh_name} (group {group['index']})")


                        blender_mesh.from_pydata(positions, [], blender_faces)
                        blender_mesh.update()

                        # --- Assign Vertex Weights ---
                        vgroup_map = {}
                        for v_idx, (b_indices, b_weights) in enumerate(zip(bone_indices, bone_weights)):
                            for b, w in zip(b_indices, b_weights):
                                if b >= 0:
                                    vg = vgroup_map.get(b)
                                    if vg is None:
                                        vg = blender_object.vertex_groups.new(name=f"Bone_{b}")
                                        vgroup_map[b] = vg
                                    vg.add([v_idx], w, 'REPLACE')

                        # --- Normals ---
                        loop_normals = []
                        for loop in blender_mesh.loops:
                            vidx = loop.vertex_index
                            loop_normals.append(Vector(normals[vidx]).normalized())

                        for poly in blender_mesh.polygons:
                            poly.use_smooth = True

                        blender_mesh.normals_split_custom_set(loop_normals)

                        # --- UVs ---
                        for i, uv_id in enumerate(UV_COMPONENT_IDS):
                            if any(v != (0.0, 0.0) for v in uvs_data[uv_id]):
                                uv_layer = blender_mesh.uv_layers.new(name=f"UVMap_{i}")
                                for loop in blender_mesh.loops:
                                    uv_layer.data[loop.index].uv = uvs_data[uv_id][loop.vertex_index]

                        # --- Vertex Colors ---
                        for i, col_id in enumerate(COLOR_COMPONENT_IDS):
                            if any(c != (1.0, 1.0, 1.0, 1.0) for c in color_data[col_id]):
                                color_layer = blender_mesh.color_attributes.new(name=f"Color_{i}", type='BYTE_COLOR', domain='CORNER')
                                for loop in blender_mesh.loops:
                                    color_layer.data[loop.index].color = color_data[col_id][loop.vertex_index]

                        blender_mesh.update()


            self.report({'INFO'}, "All position components imported as meshes.")
            return {'FINISHED'}

        except Exception as e:
            self.report({'ERROR'}, f"Error: {e}")
            return {'CANCELLED'}

    def read_post_header_id(self, f):
        # Read initial Section Header + Relocations
        magic, size = struct.unpack('<II', f.read(8))
        type_byte, skip_byte = struct.unpack('<BB', f.read(2))
        version_id = struct.unpack('<H', f.read(2))[0]
        packed_data = struct.unpack('<I', f.read(4))[0]
        relocation_table_size = (packed_data >> 8) & 0xFFFFFF
        f.read(8)  # id + specMask

        if relocation_table_size > 0:
            num_relocations = struct.unpack('<5I', f.read(20))
            for i in range(num_relocations[0]):
                f.read(8)
            for count, size in zip(num_relocations[1:], [8, 4, 4, 4]):
                for _ in range(count):
                    f.read(size)

        post_header_int = struct.unpack('<I', f.read(4))[0]
        print(f"Post-header integer: {post_header_int}")
        return post_header_int

    def read_section_header_and_find_offset(self, f, target_offset=None):
        magic, size = struct.unpack('<II', f.read(8))
        type_byte, skip_byte = struct.unpack('<BB', f.read(2))
        version_id = struct.unpack('<H', f.read(2))[0]
        packed_data = struct.unpack('<I', f.read(4))[0]
        relocation_table_size = (packed_data >> 8) & 0xFFFFFF
        f.read(8)  # id + specMask

        found_ref_offset = None
        if relocation_table_size > 0:
            num_relocations = struct.unpack('<5I', f.read(20))
            for i in range(num_relocations[0]):
                offset, ref_offset = struct.unpack('<II', f.read(8))
                if target_offset is not None and offset == target_offset and found_ref_offset is None:
                    found_ref_offset = ref_offset
            for count, size in zip(num_relocations[1:], [8, 4, 4, 4]):
                for _ in range(count):
                    f.read(size)

        return f.tell(), found_ref_offset

    def find_file_by_id(self, directory, id_hex):
        for file in directory.iterdir():
            if file.is_file() and f"_{id_hex}" in file.stem:
                return file
        return None

    def invoke(self, context, event):
        self.filter_glob = "*.obj"
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}
    
class ExportUnderworldModel(bpy.types.Operator):
    bl_idname = "tru.export_trugol_model"
    bl_label = "Export TRU Model"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        self.report({'INFO'}, "Export TRU Model (not yet implemented)")
        return {'FINISHED'}
    
class ImportUnderworldGOLAnimation(bpy.types.Operator):
    bl_idname = "tru.import_trugol_animation"
    bl_label = "Import TRU Animation"
    bl_description = "Import skeletal animation from custom .ani format"

    filepath: bpy.props.StringProperty(subtype='FILE_PATH')
    filter_glob: bpy.props.StringProperty(default="*.ani", options={'HIDDEN'})
    files: bpy.props.CollectionProperty(type=bpy.types.PropertyGroup)  # Enable multi-file selection

    def execute(self, context):
        if not context.active_object or context.active_object.type != 'ARMATURE':
            self.report({'ERROR'}, "Select an Armature to import animation onto.")
            return {'CANCELLED'}

        armature = context.active_object

        # Collect selected files
        directory = Path(self.filepath).parent
        file_paths = [directory / f.name for f in self.files]

        if not file_paths:
            file_paths = [Path(self.filepath)]

        imported = 0
        errors = []

        for path in file_paths:
            if not path.exists():
                errors.append(f"File not found: {path}")
                continue

            try:
                with open(path, "rb") as f:
                    data = f.read()

                magic = data[:4]
                if magic == b'SECT':
                    endian = '<'
                    print("Detected Little Endian format (SECT)")
                elif magic == b'TCES':
                    endian = '>'
                    print("Detected Big Endian format (TCES)")
                else:
                    errors.append(f"Unknown file format: {magic}")
                    continue

                anim_id = struct.unpack_from(f"{endian}h", data, 0x10)[0]
                key_count = struct.unpack_from(f"{endian}h", data, 0x4a)[0]
                time_per_key = struct.unpack_from(f"{endian}h", data, 0x4c)[0]
                segment_count = struct.unpack_from(f"{endian}B", data, 0x4e)[0]
                mSectionCount = struct.unpack_from(f"{endian}B", data, 0x4f)[0]
                mSectionDataOffset = struct.unpack_from(f"{endian}I", data, 0x5c)[0]
                track_data_offset = 0x60
                transform_flags = data[track_data_offset]
                final_frame = key_count - 1

                def parse_track_flags(data, offset, transform_flags, segment_count):
                    flags_bits = int.from_bytes(data[offset:offset + 512], 'little')
                    bit_pos = 0
                    rot_flags, scale_flags, pos_flags = [], [], []

                    def read_3bit_flags():
                        nonlocal bit_pos
                        flags = []
                        for _ in range(segment_count):
                            flags.append((flags_bits >> bit_pos) & 0b111)
                            bit_pos += 3
                        return flags

                    def align_byte():
                        nonlocal bit_pos
                        if bit_pos % 8 != 0:
                            bit_pos += 8 - (bit_pos % 8)

                    if transform_flags & 0b001:
                        rot_flags = read_3bit_flags()
                        align_byte()
                    if transform_flags & 0b010:
                        scale_flags = read_3bit_flags()
                        align_byte()
                    if transform_flags & 0b100:
                        pos_flags = read_3bit_flags()
                        align_byte()

                    return rot_flags, scale_flags, pos_flags

                def flags_to_axes(flag):
                    return [axis for axis, bit in zip("xyz", [1, 2, 4]) if flag & bit]

                track_flags_offset = track_data_offset + 1
                rot_flags, scale_flags, pos_flags = parse_track_flags(
                    data, track_flags_offset, transform_flags, segment_count
                )

                channel_map = []
                for i in range(segment_count):
                    channel_map.append({
                        "rotation": flags_to_axes(rot_flags[i]) if rot_flags else [],
                        "scale": flags_to_axes(scale_flags[i]) if scale_flags else [],
                        "location": flags_to_axes(pos_flags[i]) if pos_flags else [],
                    })

                stream = BytesIO(data)
                track_data_start = mSectionDataOffset + 92
                stream.seek(track_data_start)

                def read_track_data(stream, key_count):
                    header = stream.read(4)
                    if len(header) < 4:
                        return []
                    mode, count = struct.unpack(f"{endian}HH", header)

                    if mode == 2 and count > 0:
                        times = [0]
                        acc = 0
                        for _ in range(count - 1):
                            if stream.tell() >= len(data):
                                return []
                            delta = struct.unpack(f"{endian}B", stream.read(1))[0]
                            acc += delta
                            times.append(acc)
                        if stream.tell() % 4 != 0:
                            stream.seek((stream.tell() + 3) & ~3)
                        abs_val = 0.0
                        keys = []
                        for t in times:
                            if stream.tell() + 4 > len(data):
                                break
                            delta = struct.unpack(f"{endian}f", stream.read(4))[0]
                            abs_val += delta
                            keys.append((t, abs_val))
                        return keys

                    elif mode == 1:
                        if stream.tell() + 4 > len(data):
                            return []
                        val = struct.unpack(f"{endian}f", stream.read(4))[0]
                        return [(0, val)]

                    elif mode == 0:
                        keys = []
                        for i in range(key_count):
                            if stream.tell() + 4 > len(data):
                                break
                            val = struct.unpack(f"{endian}f", stream.read(4))[0]
                            keys.append((i, val))
                        return keys

                    else:
                        return []

                all_tracks = []
                for bone_index, bone_flags in enumerate(channel_map):
                    bone_tracks = []
                    for kind in ["rotation", "scale", "location"]:
                        for axis in bone_flags[kind]:
                            keyframes = read_track_data(stream, key_count)
                            bone_tracks.append((kind, axis, keyframes))
                    all_tracks.append(bone_tracks)

                base_name = f"{path.stem}"
                action_name = base_name
                suffix = 1
                while action_name in bpy.data.actions:
                    action_name = f"{base_name}_{suffix}"
                    suffix += 1
                action = bpy.data.actions.new(name=action_name)
                action["time_per_key"] = time_per_key
                action.tr7ae_anim_settings.time_per_key = time_per_key
                action["anim_id"] = anim_id
                action.tr7ae_anim_settings.anim_id = anim_id

                armature.animation_data_create()
                armature.animation_data.action = action

                context.scene.frame_set(0)

                animated_flags = {
                    bone.name: {"location": set(), "rotation": set(), "scale": set()}
                    for bone in armature.pose.bones
                }

                for bone in armature.pose.bones:
                    bone_name = bone.name
                    flags = animated_flags[bone_name]

                    if "x" not in flags["location"]:
                        bone.location.x = 0.0
                    if "y" not in flags["location"]:
                        bone.location.y = 0.0
                    if "z" not in flags["location"]:
                        bone.location.z = 0.0

                    if "x" not in flags["rotation"]:
                        bone.rotation_euler.x = 0.0
                    if "y" not in flags["rotation"]:
                        bone.rotation_euler.y = 0.0
                    if "z" not in flags["rotation"]:
                        bone.rotation_euler.z = 0.0

                    if "x" not in flags["scale"]:
                        bone.scale.x = 1.0
                    if "y" not in flags["scale"]:
                        bone.scale.y = 1.0
                    if "z" not in flags["scale"]:
                        bone.scale.z = 1.0

                for bone_index in range(len(armature.pose.bones)):
                    armature.pose.bones[bone_index].rotation_mode = 'XYZ'

                for bone_index, tracks in enumerate(all_tracks):
                    bone_name = f"Bone_{bone_index}"
                    bone = armature.pose.bones.get(bone_name)
                    if bone is None:
                        print(f"Bone '{bone_name}' not found in armature, skipping.")
                        continue

                    for kind, axis, keyframes in tracks:
                        if not keyframes:
                            continue
                        idx = "xyz".index(axis)

                        if kind == "location":
                            fcurve = action.fcurves.new(data_path=f'pose.bones["{bone_name}"].location', index=idx)
                            for frame, value in keyframes:
                                vec = mathutils.Vector((0.0, 0.0, 0.0))
                                vec[idx] = value
                                mat = mathutils.Matrix.Translation(vec)
                                if bone.parent:
                                    mat = bone.parent.bone.matrix_local @ bone.bone.matrix_local.inverted() @ mat
                                else:
                                    mat = bone.bone.matrix_local.inverted() @ mat
                                local_vec = mat.to_translation()
                                fcurve.keyframe_points.insert(frame=frame, value=local_vec[idx])
                        else:
                            path = f'pose.bones["{bone_name}"].{"rotation_euler" if kind=="rotation" else kind}'
                            fcurve = action.fcurves.new(data_path=path, index=idx)
                            for frame, value in keyframes:
                                fcurve.keyframe_points.insert(frame=frame, value=value)

                context.scene.frame_start = 0
                context.scene.frame_end = final_frame
                context.scene.frame_current = 0

                imported += 1
                self.report({'INFO'}, f"Imported animation '{action.name}'.")

            except Exception as e:
                errors.append(f"Failed to import {path.name}: {e}")

        if imported == 0:
            self.report({'ERROR'}, f"No animations imported. {', '.join(errors)}")
            return {'CANCELLED'}

        self.report({'INFO'}, f"Imported {imported} animations. {len(errors)} errors.")
        return {'FINISHED'}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    
class ExportUnderworldAnimation(bpy.types.Operator):
    bl_idname = "tru.export_trugol_animation"
    bl_label = "Export TRU Animation"
    bl_options = {'REGISTER', 'UNDO'}

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    filter_glob: bpy.props.StringProperty(default="*.ani", options={'HIDDEN'})

    def execute(self, context):
        armature = context.active_object
        if not armature or armature.type != 'ARMATURE':
            self.report({'ERROR'}, "Select an armature to export from.")
            return {'CANCELLED'}

        if not armature.animation_data or not armature.animation_data.action:
            self.report({'ERROR'}, "No active action found on armature.")
            return {'CANCELLED'}

        active_action = armature.animation_data.action
        try:
            # Prepare tracks and quaternion storage
            bone_names = [f"Bone_{i}" for i in range(len(armature.pose.bones))]
            bone_tracks = {name: {'location': [[], [], []], 'rotation': [[], [], []], 'scale': [[], [], []]} for name in bone_names}
            quaternion_frames = {name: {} for name in bone_names}
            keyframe_set = set()

            # Collect raw keyframe data
            for fcurve in active_action.fcurves:
                path = fcurve.data_path
                if not path.startswith("pose.bones"): continue

                bone_name = path.split('"')[1]
                prop = path.split('.')[-1]
                idx = fcurve.array_index

                # Gather quaternion components
                if prop == 'rotation_quaternion':
                    for kp in fcurve.keyframe_points:
                        frame = int(kp.co[0]); value = kp.co[1]
                        keyframe_set.add(frame)
                        qdict = quaternion_frames[bone_name].setdefault(frame, [None]*4)
                        qdict[idx] = value
                    continue

                # Map other props
                prop_map = {'location': 'location', 'scale': 'scale', 'rotation_euler': 'rotation'}
                if prop not in prop_map: continue
                mapped = prop_map[prop]

                for kp in fcurve.keyframe_points:
                    frame = int(kp.co[0]); value = kp.co[1]
                    keyframe_set.add(frame)

                    if mapped == 'location':
                        pb = armature.pose.bones[bone_name]
                        vec = mathutils.Vector((0.0, 0.0, 0.0))
                        vec[idx] = value
                        mat = mathutils.Matrix.Translation(vec)
                        if pb.parent:
                            mat = pb.bone.matrix_local @ pb.parent.bone.matrix_local.inverted() @ mat
                        else:
                            mat = pb.bone.matrix_local @ mat
                        value = mat.to_translation()[idx]

                    bone_tracks[bone_name][mapped][idx].append((frame, value))

            # Convert quaternions to Euler per-axis
            for bone_name, frames in quaternion_frames.items():
                for frame, comps in frames.items():
                    if None in comps: continue
                    quat = mathutils.Quaternion(comps)
                    euler = quat.to_euler('XYZ')
                    for ax in range(3):
                        bone_tracks[bone_name]['rotation'][ax].append((frame, euler[ax]))

            # Frame setup
            frame_start, frame_end = map(int, active_action.frame_range)
            keyframes = list(range(frame_start, frame_end + 1))
            key_count = len(keyframes)
            frame_index_map = {f: i for i, f in enumerate(keyframes)}
            time_per_key = int(active_action.tr7ae_anim_settings.time_per_key)
            anim_id = int(active_action.tr7ae_anim_settings.anim_id)

            # Channel flags
            segment_count = len(bone_names)
            channel_flags = {'rotation': [], 'scale': [], 'location': []}
            for bn in bone_names:
                for kind in channel_flags:
                    flag = 0
                    for ax in range(3):
                        if bone_tracks[bn][kind][ax]: flag |= (1 << ax)
                    channel_flags[kind].append(flag)

            transform_flags = 0
            if any(channel_flags['rotation']): transform_flags |= 0b001
            if any(channel_flags['scale']):    transform_flags |= 0b010
            if any(channel_flags['location']): transform_flags |= 0b100

            def pack_3bit_flags(flags_list):
                packed = bytearray(); acc = 0; bitpos = 0
                for f in flags_list:
                    acc |= ((f & 0b111) << bitpos)
                    bitpos += 3
                    while bitpos >= 8:
                        packed.append(acc & 0xFF)
                        acc >>= 8; bitpos -= 8
                if bitpos: packed.append(acc & 0xFF)
                return packed

            # --- Header ---
            header = bytearray(b'\x00'*0x30)
            header += struct.pack("<h", anim_id)
            header += struct.pack("<h", key_count)
            header += struct.pack("<h", time_per_key)
            header += struct.pack("<B", segment_count)
            header += struct.pack("<B", 1)
            header += struct.pack("<I", 0)
            header += struct.pack("<I", 0)
            header += struct.pack("<I", 0)

            sec_off_pos = len(header)
            header += struct.pack("<I", 0xDEADBEEF)
            header += struct.pack("<B", transform_flags)

            track_flags = bytearray()
            for kind, mask in [('rotation',0b001), ('scale',0b010), ('location',0b100)]:
                if transform_flags & mask:
                    track_flags += pack_3bit_flags(channel_flags[kind])
            track_flags += b'\x00' * max(0, 64 - len(track_flags))
            header += track_flags

            pad = (-len(header)) % 4
            if pad: header += b'\x00'*pad
            aligned = len(header)
            header[sec_off_pos:sec_off_pos+4] = struct.pack("<I", aligned - 68)

            binary = bytearray(header)

            # --- Track Data ---
            masks = {'rotation':0b001,'scale':0b010,'location':0b100}
            for bi, bn in enumerate(bone_names):
                for kind in ('rotation','scale','location'):
                    if not (transform_flags & masks[kind]): continue
                    af = channel_flags[kind][bi]
                    for ax in range(3):
                        if not (af & (1<<ax)): continue
                        channel = sorted(bone_tracks[bn][kind][ax])
                        kf_cnt = len(channel)

                        # Mode 0: full range
                        if kf_cnt == key_count and all(frame_index_map[f] == i for i,(f,_) in enumerate(channel)):
                            binary += struct.pack("<HH", 0, key_count)
                            for _,v in channel:
                                binary += struct.pack("<f", v)
                            continue

                        # Mode 1: constant
                        if all(v == channel[0][1] for _,v in channel):
                            binary += struct.pack("<HH", 1, 0)
                            binary += struct.pack("<f", channel[0][1])
                            continue

                        # Mode 2: sparse deltas
                        frames = [f for f,_ in channel]
                        values = [v for _,v in channel]
                        deltas = [frames[i] - frames[i-1] for i in range(1, kf_cnt)]

                        binary += struct.pack("<HH", 2, kf_cnt)
                        binary += struct.pack(f"<{len(deltas)}B", *deltas)
                        pad2 = (-len(deltas)) % 4
                        if pad2: binary += b'\x00'*pad2

                        prev = values[0]
                        binary += struct.pack("<f", prev)
                        for v in values[1:]:
                            delta_v = v - prev
                            binary += struct.pack("<f", delta_v)
                            prev = v

            # --- Section Header ---
            def write_sect_header(data_len, m_type=2, sid=1):
                return (struct.pack("<I", 0x54434553) + struct.pack("<I", data_len)
                        + struct.pack("<I", m_type) + struct.pack("<I", 0)
                        + struct.pack("<I", anim_id) + struct.pack("<I", 0xFFFFFFFF))

            sect = write_sect_header(len(binary))
            with open(self.filepath, 'wb') as f:
                f.write(sect)
                f.write(binary)

            self.report({'INFO'}, f"Exported action '{active_action.name}' with {key_count} frames.")
            return {'FINISHED'}

        except Exception as e:
            self.report({'ERROR'}, f"Export failed: {e}")
            return {'CANCELLED'}

    def invoke(self, context, event):
        self.filepath = bpy.path.ensure_ext(bpy.data.filepath, ".ani")
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

class TR7AE_OT_NormalizeAndLimitWeights(bpy.types.Operator):
    bl_idname = "tr7ae.normalize_and_limit_weights"
    bl_label = "Limit Weights and Normalize All"
    bl_description = "Automatically limit all weights per vertex to 2 and normalize them"

    def execute(self, context):
        obj = context.active_object

        if obj and obj.type == 'ARMATURE':
            mesh_objs = [child for child in obj.children if child.type == 'MESH']
        elif obj and obj.type == 'MESH':
            mesh_objs = [obj]
        else:
            self.report({'ERROR'}, "Select an armature or mesh.")
            return {'CANCELLED'}

        for mesh in mesh_objs:
            bpy.context.view_layer.objects.active = mesh
            bpy.ops.object.mode_set(mode='WEIGHT_PAINT')
            bpy.ops.object.vertex_group_normalize_all()
            bpy.ops.object.vertex_group_limit_total(limit=2)
            bpy.ops.object.vertex_group_clean(
                group_select_mode='ALL',
                limit=0.0001,
                keep_single=False
            )
            bpy.ops.object.mode_set(mode='OBJECT')

        self.report({'INFO'}, f"Processed {len(mesh_objs)} mesh object(s).")
        return {'FINISHED'}

class TR7AE_PT_UtilitiesPanel(bpy.types.Panel):
    bl_label = "Utilities"
    bl_idname = "TR7AE_PT_utilities_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'TRLAU'
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        if not obj or obj.type not in {'MESH', 'ARMATURE'}:
            return False

        tr7_type = obj.get("tr7ae_type", "")
        excluded = {"HSphere", "HMarker", "HBox", "HCapsule", "Target", "MarkUp"}
        return tr7_type not in excluded

    def draw(self, context):
        layout = self.layout
        layout.operator("tr7ae.normalize_and_limit_weights")

class TR7AE_OT_ToggleHBoxes(bpy.types.Operator):
    bl_idname = "tr7ae.toggle_hboxes"
    bl_label = "Toggle HBoxes Visibility"
    bl_description = "Hide or show all HBox objects in the scene"

    def execute(self, context):
        hboxes = [obj for obj in bpy.data.objects if obj.name.startswith("HBox_")]
        if not hboxes:
            self.report({'WARNING'}, "No HBoxes found.")
            return {'CANCELLED'}

        should_hide = not hboxes[0].hide_get()
        for obj in hboxes:
            obj.hide_set(should_hide)

        self.report({'INFO'}, f"{'Hidden' if should_hide else 'Shown'} {len(hboxes)} HBox(es)")
        return {'FINISHED'}


class TR7AE_OT_ToggleHMarkers(bpy.types.Operator):
    bl_idname = "tr7ae.toggle_hmarkers"
    bl_label = "Toggle HMarkers Visibility"
    bl_description = "Hide or show all HMarker objects in the scene"

    def execute(self, context):
        hmarkers = [obj for obj in bpy.data.objects if obj.name.startswith("HMarker_")]
        if not hmarkers:
            self.report({'WARNING'}, "No HMarkers found.")
            return {'CANCELLED'}

        should_hide = not hmarkers[0].hide_get()
        for obj in hmarkers:
            obj.hide_set(should_hide)

        self.report({'INFO'}, f"{'Hidden' if should_hide else 'Shown'} {len(hmarkers)} HMarker(s)")
        return {'FINISHED'}


class TR7AE_OT_ToggleHSpheres(bpy.types.Operator):
    bl_idname = "tr7ae.toggle_hspheres"
    bl_label = "Toggle HSpheres Visibility"
    bl_description = "Hide or show all HSphere objects in the scene"

    def execute(self, context):
        hspheres = [obj for obj in bpy.data.objects if obj.name.startswith("HSphere_")]
        if not hspheres:
            self.report({'WARNING'}, "No HSpheres found.")
            return {'CANCELLED'}

        should_hide = not hspheres[0].hide_get()

        for obj in hspheres:
            obj.hide_set(should_hide)

        action = "Hidden" if should_hide else "Shown"
        self.report({'INFO'}, f"{action} {len(hspheres)} HSphere(s)")
        return {'FINISHED'}
    
class TR7AE_OT_ToggleHCapsules(bpy.types.Operator):
    bl_idname = "tr7ae.toggle_hcapsules"
    bl_label = "Toggle HCapsules Visibility"
    bl_description = "Hide or show all HCapsules objects in the scene"

    def execute(self, context):
        hcapsules = [obj for obj in bpy.data.objects if obj.name.startswith("HCapsule_")]
        if not hcapsules:
            self.report({'WARNING'}, "No HCapsules found.")
            return {'CANCELLED'}

        should_hide = not hcapsules[0].hide_get()

        for obj in hcapsules:
            obj.hide_set(should_hide)

        action = "Hidden" if should_hide else "Shown"
        self.report({'INFO'}, f"{action} {len(hcapsules)} HCapsules(s)")
        return {'FINISHED'}
    
class TR7AE_OT_ClearTextureCache(bpy.types.Operator):
    bl_idname = "tr7ae.clear_texture_cache"
    bl_label = "Clear Texture Cache"
    bl_description = "Delete all files in the TRLAU texture cache folder"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        # Match your existing texture_dir definition
        cache_dir = os.path.join(
            bpy.utils.user_resource('SCRIPTS'),
            "addons",
            "trlau_textures"
        )

        if not os.path.exists(cache_dir):
            self.report({'WARNING'}, f"Texture cache folder not found.")
            return {'CANCELLED'}

        deleted = 0
        for fname in os.listdir(cache_dir):
            fpath = os.path.join(cache_dir, fname)
            if os.path.isfile(fpath):
                try:
                    os.remove(fpath)
                    deleted += 1
                except Exception as e:
                    self.report({'WARNING'}, f"Failed to delete {fname}: {e}")

        self.report({'INFO'}, f"Deleted {deleted} files from texture cache.")
        return {'FINISHED'}

def draw_section(layout, title, icon):
    box = layout.box()
    row = box.row()
    row.alignment = 'CENTER'
    row.label(text=title, icon=icon)
    return box

class TR7AE_PT_Tools(Panel):
    bl_label = "Tomb Raider LAU Mesh Editor"
    bl_idname = "TR7AE_PT_Tools"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'TRLAU'

    _sphere_expand = {}
    _box_expand = {}
    _marker_expand = {}
    _capsule_expand = {}

    def draw(self, context):
        layout = self.layout

        # Model Import
        box = draw_section(layout, "Model Import", 'IMPORT')
        box.operator(TR7AE_OT_ImportModel.bl_idname, text="Import TR7AE Model", icon='MESH_DATA')
        box.operator("import_scene.tr7aemeshps2", text="Import TR7AE PS2 Model", icon='MESH_DATA')
        box.operator("import_scene.pbrwc", text="Import TR7AE PS3 Model", icon='MESH_DATA')
        box.operator("tr7ae.import_nextgen_model", text="Import TR7AE Next Gen Model", icon='MESH_DATA')
        box.operator("tru.import_trugol_model", text="Import TRUGOL Model", icon='MESH_DATA')

        # Model Export (conditional)
        if context.active_object and context.active_object.type == 'ARMATURE':
            box = draw_section(layout, "Model Export", 'EXPORT')
            box.operator("tr7ae.export_oldgen_model", text="Export TR7AE Model", icon='EXPORT')
            box.operator("tr7ae.export_nextgen_model", text="Export TR7AE Next Gen Model", icon='EXPORT')
            box.operator("tru.export_trugol_model", text="Export TRU Model", icon='EXPORT')

        # Level Import
        box = draw_section(layout, "Level Import", 'WORLD_DATA')
        box.operator("tr7ae.import_level", text="Import TR7AE Level", icon='WORLD')

        # Animation (conditional)
        if context.active_object and context.active_object.type == 'ARMATURE':
            box = draw_section(layout, "Animation Import", 'ANIM')
            box.operator("tr7ae.import_animation", text="Import TR7AE Animation", icon='ARMATURE_DATA')
            box.operator("tru.import_trugol_animation", text="Import TRU Animation", icon='ARMATURE_DATA')

            box = draw_section(layout, "Animation Export", 'ANIM')
            box.operator("tr7ae.export_animation", text="Export TR7AE Animation", icon='ARMATURE_DATA')
            box.operator("tru.export_trugol_animation", text="Export TRU Animation", icon='ARMATURE_DATA')

        action = context.object.animation_data.action if context.object and context.object.animation_data else None
        if action:
            layout.label(text="Animation Settings")
            layout.prop(action.tr7ae_anim_settings, "time_per_key")
            layout.prop(action.tr7ae_anim_settings, "anim_id")

        obj = context.active_object
        if obj and obj.get("tr7ae_type") == "ModelTarget":
            box = layout.box()
            box.label(text="Model Target Properties", icon='MOD_ARMATURE')
            layout.prop(obj.tr7ae_modeltarget, "px")
            layout.prop(obj.tr7ae_modeltarget, "py")
            layout.prop(obj.tr7ae_modeltarget, "pz")
            layout.prop(obj.tr7ae_modeltarget, "rx")
            layout.prop(obj.tr7ae_modeltarget, "ry")
            layout.prop(obj.tr7ae_modeltarget, "rz")
            box.prop(obj.tr7ae_modeltarget, "flags")
            box.prop(obj.tr7ae_modeltarget, "unique_id")


        obj = context.active_object
        if obj and obj.get("tr7ae_is_mface"):
            layout.operator("tr7ae.export_modified_mface", icon='EXPORT')

        bone = context.active_pose_bone
        if not bone:
            return
    
def is_image_fully_opaque_blender(image_path):
    try:
        img = bpy.data.images.load(str(image_path), check_existing=False)
        has_alpha = img.channels == 4
        fully_opaque = True

        if has_alpha:
            pixels = list(img.pixels)  # RGBA per pixel
            for i in range(3, len(pixels), 4):
                if pixels[i] < 0.999:
                    fully_opaque = False
                    break

        bpy.data.images.remove(img)
        return fully_opaque
    except:
        return False
        
class TR7AE_OT_ConvertImageToPCD(bpy.types.Operator):
    bl_idname = "tr7ae.convert_image_to_pcd"
    bl_label = "Convert Image(s) to PCD"
    bl_description = "Convert DDS, PNG, JPG, TGA or other image formats to PCD\n\nRequires set path to texconv.exe in addon preferences for\nany texture format that isn't DDS"

    files: CollectionProperty(type=OperatorFileListElement)
    directory: StringProperty(subtype='DIR_PATH')
    filepath: StringProperty(subtype='FILE_PATH')
    filter_glob: StringProperty(default="*.png;*.tga;*.bmp;*.jpg;*.jpeg;*.gif;*.tif;*.tiff;*.hdr;*.ppm", options={'HIDDEN'})

    def execute(self, context):
        try:
            targets = self.files or [Path(self.filepath).name]

            for f in targets:
                ext = f.name.lower()
                full_path = Path(self.directory) / f.name
                if not full_path.exists():
                    continue

                if ext.endswith(".dds"):
                    temp_dds = full_path
                else:
                    temp_dds = Path(tempfile.gettempdir()) / (full_path.stem + ".dds")
                    prefs = context.preferences.addons[__name__].preferences
                    texconv_path = getattr(prefs, 'texconv_path', '') or "texconv.exe"
                    format = "DXT1" if is_image_fully_opaque_blender(full_path) else "DXT5"

                    if not os.path.exists(texconv_path):
                        raise RuntimeError(f"Cannot convert non-DDS textures without a valid path to texconv.exe.\n\nSet a valid texconv.exe path in the TRLAU Mesh Editor Addon Preferences.")

                    if not input_path.exists():
                        raise RuntimeError(f"Texture file does not exist: {input_path}")

                    try:
                        result = subprocess.run([
                            texconv_path,
                            "-nologo",
                            "-y",
                            "-ft", "dds",
                            "-f", format,
                            "-o", str(temp_dds.parent),
                            str(input_path)
                        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                    except FileNotFoundError:
                        raise RuntimeError(f"Failed to launch texconv.exe: File not found.\nMake sure texconv.exe is accessible at: {texconv_path}")
                    except subprocess.CalledProcessError as e:
                        stderr = e.stderr.decode().strip()
                        stdout = e.stdout.decode().strip()
                        raise RuntimeError(f"texconv failed:\n{stderr or '(no stderr output)'}\n{stdout or ''}")

                    if not temp_dds.exists():
                        raise RuntimeError(f"Expected output DDS was not created: {temp_dds}")


                output_pcd_path = full_path.with_suffix(".pcd")
                convert_dds_to_pcd(temp_dds, output_pcd_path)

            self.report({'INFO'}, "Texture conversion complete.")
            return {'FINISHED'}

        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}

    def invoke(self, context, event):
        self.filter_glob = "*.dds;*.png;*.tga;*.bmp;*.jpg;*.jpeg;*.gif;*.tif;*.tiff;*.hdr;*.ppm"
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}
    
class TR7AE_OT_ConvertPCDToImage(bpy.types.Operator):
    bl_idname = "tr7ae.convert_pcd_to_image"
    bl_label = "Convert PCD to Image"
    bl_description = "Convert PCD texture files to DDS, PNG, TGA, JPG, or BMP using texconv"

    files: CollectionProperty(type=OperatorFileListElement)
    directory: StringProperty(subtype='DIR_PATH')
    filepath: StringProperty(subtype='FILE_PATH')
    filter_glob: StringProperty(default="*.pcd", options={'HIDDEN'})
    format: EnumProperty(
        name="Format",
        items=[
            ('PNG', "PNG", ""),
            ('JPEG', "JPG", ""),
            ('TGA', "TGA", ""),
            ('BMP', "BMP", ""),
            ('DDS', "DDS", ""),
        ],
        default='DDS'
    )

    def execute(self, context):
        import struct, tempfile, subprocess
        from pathlib import Path

        prefs = context.preferences.addons[__name__].preferences
        texconv_path = getattr(prefs, 'texconv_path', '') or "texconv.exe"

        def extract_dds_from_pcd(pcd_path):
            with open(pcd_path, "rb") as f:
                f.seek(28)
                format_bytes = f.read(4)
                dds_format = struct.unpack('<I', format_bytes)[0]

                f.seek(24)
                texture_header = f.read(24)
                (
                    magic_number,
                    _,
                    bitmap_size,
                    _,
                    width,
                    height,
                    _,
                    mipmaps,
                    _
                ) = struct.unpack("<I i I I H H B B H", texture_header)

                if magic_number != 0x39444350:
                    raise ValueError(f"Unexpected magic number: {hex(magic_number)}")

                f.seek(48)
                dxt_data = f.read(bitmap_size)

                if len(dxt_data) < bitmap_size:
                    raise ValueError(f"Truncated data: expected {bitmap_size}, got {len(dxt_data)}")

                def create_dds_header(width, height, mipmaps, dds_format, dxt_size):
                    header = bytearray(128)
                    struct.pack_into('<4sI', header, 0, b'DDS ', 124)
                    struct.pack_into('<I', header, 8, 0x0002100F)
                    struct.pack_into('<I', header, 12, height)
                    struct.pack_into('<I', header, 16, width)
                    struct.pack_into('<I', header, 20, max(1, dxt_size))
                    struct.pack_into('<I', header, 28, mipmaps if mipmaps > 0 else 1)

                    struct.pack_into('<I', header, 76, 32)
                    struct.pack_into('<I', header, 80, 0x00000004)
                    struct.pack_into('<4s', header, 84, struct.pack("<I", dds_format))  # DXT1/5
                    struct.pack_into('<I', header, 108, 0x1000)
                    return header

                dds_header = create_dds_header(width, height, mipmaps, dds_format, len(dxt_data))
                return dds_header + dxt_data


        targets = self.files or [Path(self.filepath).name]
        for f in targets:
            pcd_path = Path(self.directory) / f.name
            if not pcd_path.exists():
                continue

            try:
                dds_bytes = extract_dds_from_pcd(pcd_path)

                temp_dds = Path(tempfile.gettempdir()) / f"{pcd_path.stem}.dds"
                with open(temp_dds, "wb") as out:
                    out.write(dds_bytes)

                output_ext = self.format.lower()
                output_path = pcd_path.with_suffix(f".{output_ext}")

                result = subprocess.run([
                    texconv_path,
                    "-nologo",
                    "-y",
                    "-ft", output_ext,
                    "-o", str(output_path.parent),
                    str(temp_dds)
                ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                if result.returncode != 0:
                    raise RuntimeError(result.stderr.decode() or result.stdout.decode())

                self.report({'INFO'}, f"Converted {f.name} → {output_ext.upper()}")

            except Exception as e:
                self.report({'ERROR'}, f"{f.name}: {e}")

        return {'FINISHED'}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def draw(self, context):
        self.layout.prop(self, "format")


    
class TR7AE_OT_ConvertRAWToImage(Operator):
    bl_idname = "tr7ae.convert_raw_to_image"
    bl_label = "Convert RAW(s) to Image"
    bl_description = "Convert .RAW texture file(s) to standard image formats (DDS, PNG, TGA, JPG, BMP)"

    files: CollectionProperty(type=OperatorFileListElement)
    directory: StringProperty(subtype='DIR_PATH')
    filepath: StringProperty(subtype='FILE_PATH')
    filter_glob: StringProperty(default="*.raw", options={'HIDDEN'})
    format: EnumProperty(
        name="Format",
        items=[
            ('PNG', "PNG", ""),
            ('JPEG', "JPG", ""),
            ('TARGA', "TGA", ""),
            ('BMP', "BMP", ""),
            ('DDS', "DDS", ""),
        ],
        default='PNG'
    )

    def execute(self, context):
        import struct, tempfile
        from pathlib import Path
        import bpy

        targets = self.files or [Path(self.filepath).name]

        for f in targets:
            raw_path = Path(self.directory) / f.name
            ext_map = {
                'PNG': 'png',
                'JPEG': 'jpg',
                'TARGA': 'tga',
                'BMP': 'bmp',
                'DDS': 'dds',
            }
            output_ext = ext_map.get(self.format, self.format.lower())
            output_path = raw_path.with_suffix(f".{output_ext}")
            if not raw_path.exists():
                continue

            with open(raw_path, "rb") as file:
                data = file.read()

            header = struct.unpack_from("<25i", data[:100])
            pixel_offset = header[1]
            pixel_length = header[2]
            width, height = header[5], header[6]
            depth = header[8]

            if depth not in (24, 32):
                self.report({'WARNING'}, f"Unsupported bit depth ({depth}) in {f.name}. Skipped.")
                continue

            mode = 'RGB' if depth == 24 else 'RGBA'
            channels = 3 if mode == 'RGB' else 4

            pixel_data = data[pixel_offset:pixel_offset + pixel_length]
            if len(pixel_data) < width * height * channels:
                self.report({'WARNING'}, f"Incomplete pixel data in {f.name}. Skipped.")
                continue

            img_name = raw_path.stem
            image = bpy.data.images.new(img_name, width=width, height=height, alpha=(channels == 4))
            pixels = []

            row_size = width * channels
            for y in reversed(range(height)):
                row_start = y * row_size
                for x in range(width):
                    i = row_start + x * channels
                    b = pixel_data[i] / 255.0
                    g = pixel_data[i + 1] / 255.0
                    r = pixel_data[i + 2] / 255.0
                    a = 1.0 if channels == 3 else pixel_data[i + 3] / 255.0
                    pixels.extend([r, g, b, a])

            image.pixels = pixels

            if self.format == 'DDS':
                def create_dds_header_rgba(width, height):
                    header = bytearray(128)

                    struct.pack_into('<4sI', header, 0, b'DDS ', 124)
                    struct.pack_into('<I', header, 8, 0x0002100F)
                    struct.pack_into('<I', header, 12, height)
                    struct.pack_into('<I', header, 16, width)
                    pitch = width * 4
                    struct.pack_into('<I', header, 20, pitch)

                    struct.pack_into('<I', header, 24, 0)
                    struct.pack_into('<I', header, 28, 0)

                    # Pixel Format
                    struct.pack_into('<I', header, 76, 32)
                    struct.pack_into('<I', header, 80, 0x41)
                    struct.pack_into('<4s', header, 84, b'\x00\x00\x00\x00')
                    struct.pack_into('<I', header, 88, 32)
                    struct.pack_into('<I', header, 92, 0x00FF0000)
                    struct.pack_into('<I', header, 96, 0x0000FF00)
                    struct.pack_into('<I', header, 100, 0x000000FF)
                    struct.pack_into('<I', header, 104, 0xFF000000)

                    # Caps
                    struct.pack_into('<I', header, 108, 0x1000)
                    return header

                dds_header = create_dds_header_rgba(width, height)

                with open(output_path, "wb") as dds_file:
                    dds_file.write(dds_header)
                    dds_file.write(pixel_data)

                self.report({'INFO'}, f"Saved: {output_path.name}")
            else:
                try:
                    image.filepath_raw = str(output_path)
                    image.file_format = self.format
                    image.save()
                    self.report({'INFO'}, f"Saved: {output_path.name}")
                except Exception as e:
                    self.report({'ERROR'}, f"Failed to save {output_path.name}: {e}")

            bpy.data.images.remove(image)

        return {'FINISHED'}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def draw(self, context):
        self.layout.prop(self, "format")

class TR7AE_OT_ConvertImageToRAW(bpy.types.Operator):
    bl_idname = "tr7ae.convert_image_to_raw"
    bl_label = "Convert Image(s) to RAW"
    bl_description = "Convert PNG, JPG, DDS, or other images to .RAW format"

    files: CollectionProperty(type=OperatorFileListElement)
    directory: StringProperty(subtype='DIR_PATH')
    filepath: StringProperty(subtype='FILE_PATH')
    filter_glob: StringProperty(default="*.png;*.jpg;*.jpeg;*.tga;*.bmp;*.dds", options={'HIDDEN'})

    def execute(self, context):
        import struct
        import ctypes
        from pathlib import Path

        def to_signed32(n):
            return ctypes.c_int32(n).value

        targets = self.files or [Path(self.filepath).name]

        for f in targets:
            img_path = Path(self.directory) / f.name
            if not img_path.exists():
                continue

            try:
                pixel_data = b''
                width = height = 0
                channels = 4  # Always RGBA
                is_dds = img_path.suffix.lower() == ".dds"

                if is_dds:
                    # Read compressed DDS payload directly
                    with open(img_path, "rb") as dds:
                        dds_header = dds.read(128)
                        pixel_data = dds.read()

                        # Extract width & height from DDS header
                        width = struct.unpack_from("<I", dds_header, 16)[0]
                        height = struct.unpack_from("<I", dds_header, 12)[0]
                else:
                    # Standard image via Blender
                    img = bpy.data.images.load(str(img_path), check_existing=True)
                    width, height = img.size
                    pixels = list(img.pixels)

                    # Flip vertically
                    byte_data = bytearray()
                    row_stride = width * 4
                    for y in reversed(range(height)):
                        row_start = y * row_stride
                        for x in range(width):
                            i = row_start + x * 4
                            b = int(pixels[i] * 255)
                            g = int(pixels[i + 1] * 255)
                            r = int(pixels[i + 2] * 255)
                            a = int(pixels[i + 3] * 255)
                            byte_data.extend([r, g, b, a])
                    pixel_data = bytes(byte_data)
                    bpy.data.images.remove(img)

                pixel_offset = 32 * 4
                pixel_length = len(pixel_data)

                # Build .RAW header (32 integers)
                header = struct.pack(
                    "<32i",
                    0,                        # id
                    pixel_offset,             # pixelDataStart
                    pixel_length,             # pixelDataLen
                    3, 112,                     # mode, attr
                    width, height,            # width, height
                    width * 4,                # pitch (for uncompressed)
                    32,                       # depth
                    8, 8, 8, 8,               # adepth, rdepth, gdepth, bdepth
                    to_signed32(0xFF000000),  # amask
                    to_signed32(0x00FF0000),  # rmask
                    to_signed32(0x0000FF00),  # gmask
                    to_signed32(0x000000FF),  # bmask
                    1, 0,                     # mipLevel, kValue
                    width.bit_length() - 1,
                    height.bit_length() - 1,
                    *([0] * 7),               # mipLen[7]
                    -1, -1,                     # rawVer, tgaVer
                    0,                        # alphaDepth
                    0                         # fill1
                )

                raw_path = img_path.with_suffix(".raw")
                with open(raw_path, "wb") as out:
                    out.write(header)
                    out.write(pixel_data)

                self.report({'INFO'}, f"Converted to: {raw_path.name}")

            except Exception as e:
                self.report({'ERROR'}, f"Failed to convert {f.name}: {str(e)}")

        return {'FINISHED'}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

class TR7AE_PT_TextureTools(bpy.types.Panel):
    bl_label = "Texture Tools"
    bl_idname = "TR7AE_PT_TextureTools"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'TRLAU'

    def draw(self, context):
        layout = self.layout
        layout.operator("tr7ae.convert_image_to_pcd", text="Convert Image(s) to PCD")
        layout.operator("tr7ae.convert_pcd_to_image", text="Convert PCD to Image(s)")
        row = layout.row()
        row.alignment = 'CENTER'
        row.label(text="────────────", icon='BLANK1')
        layout.operator("tr7ae.convert_image_to_raw", text="Convert Image(s) to RAW")
        layout.operator("tr7ae.convert_raw_to_image", text="Convert RAW(s) to Image")
        row = layout.row()
        row.alignment = 'CENTER'
        row.label(text="────────────", icon='BLANK1')
        layout.operator("tr7ae.clear_texture_cache", icon='TRASH')


class TR7AE_Preferences(AddonPreferences):
    bl_idname = __name__

    texconv_path: StringProperty(
        name="Path to texconv.exe",
        description="Location of the texconv executable",
        subtype='FILE_PATH',
        default=""
    )

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "texconv_path")

#Tomb Raider Legend/Anniversary Animation Importing (PC, PS2, PS3, Wii)

class TR7AE_OT_ImportAnimation(bpy.types.Operator):
    bl_idname = "tr7ae.import_animation"
    bl_label = "Import Animation"
    bl_description = "Import skeletal animation from custom .ani format"

    filepath: bpy.props.StringProperty(subtype='FILE_PATH')
    filter_glob: bpy.props.StringProperty(default="*.ani", options={'HIDDEN'})

    def execute(self, context):
        import struct
        from pathlib import Path
        from io import BytesIO
        import bpy
        import mathutils

        if not context.active_object or context.active_object.type != 'ARMATURE':
            self.report({'ERROR'}, "Select an Armature to import animation onto.")
            return {'CANCELLED'}

        armature = context.active_object
        path = Path(self.filepath)
        if not path.exists():
            self.report({'ERROR'}, "File not found.")
            return {'CANCELLED'}

        try:
            with open(path, "rb") as f:
                data = f.read()

            # Detect endianness
            magic = data[:4]
            if magic == b"SECT":
                endianness = "<"  # Little Endian
                print(f"Imported {path.name} - PC/PS2 animation")
            elif magic == b"TCES":
                endianness = ">"  # Big Endian
                print(f"Imported {path.name} - PS3/Wii animation")
            else:
                self.report({'ERROR'}, f"Unknown file signature: {magic}")
                return {'CANCELLED'}

            anim_id = struct.unpack_from(endianness + "h", data, 0x10)[0]
            key_count = struct.unpack_from(endianness + "h", data, 0x32)[0]
            time_per_key = struct.unpack_from(endianness + "h", data, 0x34)[0]
            segment_count = struct.unpack_from(endianness + "B", data, 0x36)[0]
            mSectionCount = struct.unpack_from(endianness + "B", data, 0x37)[0]
            mSectionDataOffset = struct.unpack_from(endianness + "I", data, 0x38)[0]
            track_data_offset = 0x3C
            transform_flags = data[track_data_offset]

            final_frame = key_count - 1

            def parse_track_flags(data, offset, transform_flags, segment_count):
                flags_bits = int.from_bytes(data[offset:offset + 512], 'little')
                bit_pos = 0
                rot_flags, scale_flags, pos_flags = [], [], []

                def read_3bit_flags():
                    nonlocal bit_pos
                    flags = []
                    for _ in range(segment_count):
                        flags.append((flags_bits >> bit_pos) & 0b111)
                        bit_pos += 3
                    return flags

                def align_byte():
                    nonlocal bit_pos
                    if bit_pos % 8 != 0:
                        bit_pos += 8 - (bit_pos % 8)

                if transform_flags & 0b001:
                    rot_flags = read_3bit_flags()
                    align_byte()
                if transform_flags & 0b010:
                    scale_flags = read_3bit_flags()
                    align_byte()
                if transform_flags & 0b100:
                    pos_flags = read_3bit_flags()
                    align_byte()

                return rot_flags, scale_flags, pos_flags

            def flags_to_axes(flag):
                return [axis for axis, bit in zip("xyz", [1, 2, 4]) if flag & bit]

            track_flags_offset = track_data_offset + 1
            rot_flags, scale_flags, pos_flags = parse_track_flags(
                data, track_flags_offset, transform_flags, segment_count
            )

            channel_map = []
            for i in range(segment_count):
                channel_map.append({
                    "rotation": flags_to_axes(rot_flags[i]) if rot_flags else [],
                    "scale": flags_to_axes(scale_flags[i]) if scale_flags else [],
                    "location": flags_to_axes(pos_flags[i]) if pos_flags else [],
                })

            stream = BytesIO(data)
            track_data_start = mSectionDataOffset + 56
            stream.seek(track_data_start)

            def read_track_data(stream, key_count):
                header = stream.read(4)
                if len(header) < 4:
                    return []
                mode, count = struct.unpack(endianness + "HH", header)

                if mode == 2 and count > 0:
                    times = [0]
                    acc = 0
                    for _ in range(count - 1):
                        if stream.tell() >= len(data):
                            return []
                        delta = struct.unpack(endianness + "B", stream.read(1))[0]
                        acc += delta
                        times.append(acc)
                    if stream.tell() % 4 != 0:
                        stream.seek((stream.tell() + 3) & ~3)
                    abs_val = 0.0
                    keys = []
                    for t in times:
                        if stream.tell() + 4 > len(data):
                            break
                        delta = struct.unpack(endianness + "f", stream.read(4))[0]
                        abs_val += delta
                        keys.append((t, abs_val))
                    return keys

                elif mode == 1:
                    if stream.tell() + 4 > len(data):
                        return []
                    val = struct.unpack(endianness + "f", stream.read(4))[0]
                    return [(0, val)]

                elif mode == 0:
                    keys = []
                    for i in range(key_count):
                        if stream.tell() + 4 > len(data):
                            break
                        val = struct.unpack(endianness + "f", stream.read(4))[0]
                        keys.append((i, val))
                    return keys

                else:
                    return []

            all_tracks = []
            for bone_index, bone_flags in enumerate(channel_map):
                bone_tracks = []
                for kind in ["rotation", "scale", "location"]:
                    for axis in bone_flags[kind]:
                        keyframes = read_track_data(stream, key_count)
                        bone_tracks.append((kind, axis, keyframes))
                all_tracks.append(bone_tracks)

            base_name = f"{path.stem}"
            action_name = base_name
            suffix = 1
            while action_name in bpy.data.actions:
                action_name = f"{base_name}_{suffix}"
                suffix += 1
            action = bpy.data.actions.new(name=action_name)
            action["time_per_key"] = time_per_key
            action.tr7ae_anim_settings.time_per_key = time_per_key
            action["anim_id"] = anim_id
            action.tr7ae_anim_settings.anim_id = anim_id

            armature.animation_data_create()
            armature.animation_data.action = action

            context.scene.frame_set(0)

            animated_flags = {
                bone.name: {"location": set(), "rotation": set(), "scale": set()}
                for bone in armature.pose.bones
            }

            for bone in armature.pose.bones:
                bone_name = bone.name
                flags = animated_flags[bone_name]

                if "x" not in flags["location"]:
                    bone.location.x = 0.0
                if "y" not in flags["location"]:
                    bone.location.y = 0.0
                if "z" not in flags["location"]:
                    bone.location.z = 0.0

                if "x" not in flags["rotation"]:
                    bone.rotation_euler.x = 0.0
                if "y" not in flags["rotation"]:
                    bone.rotation_euler.y = 0.0
                if "z" not in flags["rotation"]:
                    bone.rotation_euler.z = 0.0

                if "x" not in flags["scale"]:
                    bone.scale.x = 1.0
                if "y" not in flags["scale"]:
                    bone.scale.y = 1.0
                if "z" not in flags["scale"]:
                    bone.scale.z = 1.0

            for bone_index in range(len(armature.pose.bones)):
                armature.pose.bones[bone_index].rotation_mode = 'XYZ'

            for bone_index, tracks in enumerate(all_tracks):
                if bone_index >= len(armature.pose.bones):
                    continue
                bone = armature.pose.bones[bone_index]
                bone_name = bone.name
                for kind, axis, keyframes in tracks:
                    if not keyframes:
                        continue
                    idx = "xyz".index(axis)

                    if kind == "location":
                        fcurve = action.fcurves.new(data_path=f'pose.bones["{bone_name}"].location', index=idx)
                        for frame, value in keyframes:
                            vec = mathutils.Vector((0.0, 0.0, 0.0))
                            vec[idx] = value
                            mat = mathutils.Matrix.Translation(vec)
                            if bone.parent:
                                mat = bone.parent.bone.matrix_local @ bone.bone.matrix_local.inverted() @ mat
                            else:
                                mat = bone.bone.matrix_local.inverted() @ mat
                            local_vec = mat.to_translation()
                            fcurve.keyframe_points.insert(frame=frame, value=local_vec[idx])
                    else:
                        path = f'pose.bones["{bone_name}"].{"rotation_euler" if kind=="rotation" else kind}'
                        fcurve = action.fcurves.new(data_path=path, index=idx)
                        for frame, value in keyframes:
                            fcurve.keyframe_points.insert(frame=frame, value=value)

            context.scene.frame_start = 0
            context.scene.frame_end = final_frame
            context.scene.frame_current = 0

            self.report({'INFO'}, f"Imported animation '{action.name}' and set end frame to {final_frame}.")
            return {'FINISHED'}

        except Exception as e:
            self.report({'ERROR'}, f"Failed to import animation: {e}")
            return {'CANCELLED'}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    
class TR7AE_AnimationSettings(bpy.types.PropertyGroup):
    time_per_key: bpy.props.IntProperty(
        name="Time Per Key",
        description="Milliseconds per keyframe",
        default=100,
        min=1,
        max=1000
    )

    anim_id: bpy.props.IntProperty(
        name="Animation ID",
        default=1,
    )

class TR7AE_OT_ExportAnimation(bpy.types.Operator):
    bl_idname = "tr7ae.export_animation"
    bl_label = "Export Animation"
    bl_description = "Export skeletal animation to custom .ani format"

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    filter_glob: bpy.props.StringProperty(default="*.ani", options={'HIDDEN'})

    def execute(self, context):
        armature = context.active_object
        if not armature or armature.type != 'ARMATURE':
            self.report({'ERROR'}, "Select an armature to export from.")
            return {'CANCELLED'}

        if not armature.animation_data or not armature.animation_data.action:
            self.report({'ERROR'}, "No active action found on armature.")
            return {'CANCELLED'}

        active_action = armature.animation_data.action
        try:
            # Prepare tracks and quaternion storage
            bone_names = [b.name for b in armature.pose.bones]
            bone_tracks = {name: {'location': [[], [], []], 'rotation': [[], [], []], 'scale': [[], [], []]} for name in bone_names}
            quaternion_frames = {name: {} for name in bone_names}
            keyframe_set = set()

            # Collect raw keyframe data
            for fcurve in active_action.fcurves:
                path = fcurve.data_path
                if not path.startswith("pose.bones"): continue

                bone_name = path.split('"')[1]
                prop = path.split('.')[-1]
                idx = fcurve.array_index

                # Gather quaternion components
                if prop == 'rotation_quaternion':
                    for kp in fcurve.keyframe_points:
                        frame = int(kp.co[0]); value = kp.co[1]
                        keyframe_set.add(frame)
                        qdict = quaternion_frames[bone_name].setdefault(frame, [None]*4)
                        qdict[idx] = value
                    continue

                # Map other props
                prop_map = {'location': 'location', 'scale': 'scale', 'rotation_euler': 'rotation'}
                if prop not in prop_map: continue
                mapped = prop_map[prop]

                for kp in fcurve.keyframe_points:
                    frame = int(kp.co[0]); value = kp.co[1]
                    keyframe_set.add(frame)

                    if mapped == 'location':
                        pb = armature.pose.bones[bone_name]
                        vec = mathutils.Vector((0.0, 0.0, 0.0))
                        vec[idx] = value
                        mat = mathutils.Matrix.Translation(vec)
                        if pb.parent:
                            mat = pb.bone.matrix_local @ pb.parent.bone.matrix_local.inverted() @ mat
                        else:
                            mat = pb.bone.matrix_local @ mat
                        value = mat.to_translation()[idx]

                    bone_tracks[bone_name][mapped][idx].append((frame, value))

            # Convert quaternions to Euler per-axis
            for bone_name, frames in quaternion_frames.items():
                for frame, comps in frames.items():
                    if None in comps: continue
                    quat = mathutils.Quaternion(comps)
                    euler = quat.to_euler('XYZ')
                    for ax in range(3):
                        bone_tracks[bone_name]['rotation'][ax].append((frame, euler[ax]))

            # Frame setup
            frame_start, frame_end = map(int, active_action.frame_range)
            keyframes = list(range(frame_start, frame_end + 1))
            key_count = len(keyframes)
            frame_index_map = {f: i for i, f in enumerate(keyframes)}
            time_per_key = int(active_action.tr7ae_anim_settings.time_per_key)
            anim_id = int(active_action.tr7ae_anim_settings.anim_id)

            # Channel flags
            segment_count = len(bone_names)
            channel_flags = {'rotation': [], 'scale': [], 'location': []}
            for bn in bone_names:
                for kind in channel_flags:
                    flag = 0
                    for ax in range(3):
                        if bone_tracks[bn][kind][ax]: flag |= (1 << ax)
                    channel_flags[kind].append(flag)

            transform_flags = 0
            if any(channel_flags['rotation']): transform_flags |= 0b001
            if any(channel_flags['scale']):    transform_flags |= 0b010
            if any(channel_flags['location']): transform_flags |= 0b100

            def pack_3bit_flags(flags_list):
                packed = bytearray(); acc = 0; bitpos = 0
                for f in flags_list:
                    acc |= ((f & 0b111) << bitpos)
                    bitpos += 3
                    while bitpos >= 8:
                        packed.append(acc & 0xFF)
                        acc >>= 8; bitpos -= 8
                if bitpos: packed.append(acc & 0xFF)
                return packed

            # --- Header ---
            header = bytearray(b'\x00'*0x18)
            header += struct.pack("<h", anim_id)
            header += struct.pack("<h", key_count)
            header += struct.pack("<h", time_per_key)
            header += struct.pack("<B", segment_count)
            header += struct.pack("<B", 1)

            sec_off_pos = len(header)
            header += struct.pack("<I", 0xDEADBEEF)
            header += struct.pack("<B", transform_flags)

            track_flags = bytearray()
            for kind, mask in [('rotation',0b001), ('scale',0b010), ('location',0b100)]:
                if transform_flags & mask:
                    track_flags += pack_3bit_flags(channel_flags[kind])
            track_flags += b'\x00' * max(0, 64 - len(track_flags))
            header += track_flags

            pad = (-len(header)) % 4
            if pad: header += b'\x00'*pad
            aligned = len(header)
            header[sec_off_pos:sec_off_pos+4] = struct.pack("<I", aligned - 32)

            binary = bytearray(header)

            # --- Track Data ---
            masks = {'rotation':0b001,'scale':0b010,'location':0b100}
            for bi, bn in enumerate(bone_names):
                for kind in ('rotation','scale','location'):
                    if not (transform_flags & masks[kind]): continue
                    af = channel_flags[kind][bi]
                    for ax in range(3):
                        if not (af & (1<<ax)): continue
                        channel = sorted(bone_tracks[bn][kind][ax])
                        kf_cnt = len(channel)

                        # Mode 0: full range
                        if kf_cnt == key_count and all(frame_index_map[f] == i for i,(f,_) in enumerate(channel)):
                            binary += struct.pack("<HH", 0, key_count)
                            for _,v in channel:
                                binary += struct.pack("<f", v)
                            continue

                        # Mode 1: constant
                        if all(v == channel[0][1] for _,v in channel):
                            binary += struct.pack("<HH", 1, 0)
                            binary += struct.pack("<f", channel[0][1])
                            continue

                        # Mode 2: sparse deltas
                        frames = [f for f,_ in channel]
                        values = [v for _,v in channel]
                        deltas = [frames[i] - frames[i-1] for i in range(1, kf_cnt)]

                        binary += struct.pack("<HH", 2, kf_cnt)
                        binary += struct.pack(f"<{len(deltas)}B", *deltas)
                        pad2 = (-len(deltas)) % 4
                        if pad2: binary += b'\x00'*pad2

                        prev = values[0]
                        binary += struct.pack("<f", prev)
                        for v in values[1:]:
                            delta_v = v - prev
                            binary += struct.pack("<f", delta_v)
                            prev = v

            # --- Section Header ---
            def write_sect_header(data_len, m_type=2, sid=1):
                return (struct.pack("<I", 0x54434553) + struct.pack("<I", data_len)
                        + struct.pack("<I", m_type) + struct.pack("<I", 0)
                        + struct.pack("<I", anim_id) + struct.pack("<I", 0xFFFFFFFF))

            sect = write_sect_header(len(binary))
            with open(self.filepath, 'wb') as f:
                f.write(sect)
                f.write(binary)

            self.report({'INFO'}, f"Exported action '{active_action.name}' with {key_count} frames.")
            return {'FINISHED'}

        except Exception as e:
            self.report({'ERROR'}, f"Export failed: {e}")
            return {'CANCELLED'}

    def invoke(self, context, event):
        self.filepath = bpy.path.ensure_ext(bpy.data.filepath, ".ani")
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

        
class TR7AE_PT_FileSectionsPanel(bpy.types.Panel):
    bl_label = "File Section Data"
    bl_idname = "TR7AE_PT_file_sections"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'TRLAU'
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        return context.object and context.object.type == 'ARMATURE'

    def draw(self, context):
        layout = self.layout
        arm = context.object
        sections = arm.tr7ae_sections

        layout.label(text="Section Indices:")
        layout.prop(sections, "main_file_index")
        layout.prop(sections, "extra_file_index")

        if sections.cloth_file_index != 0:
            layout.prop(sections, "cloth_file_index")

        
class TR7AE_PT_ClothPanel(bpy.types.Panel):
    bl_label = "Cloth"
    bl_idname = "TR7AE_PT_Cloth"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Tomb Raider LAU Mesh Editor'
    bl_parent_id = "TR7AE_PT_Tools"

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.name == "ClothStrip"

    def draw(self, context):
        layout = self.layout
        obj = context.active_object

        layout.label(text="Cloth Setup")
        settings = obj.tr7ae_cloth
        layout.prop(settings, "gravity")
        layout.prop(settings, "drag")
        layout.prop(settings, "wind_response")
        layout.prop(settings, "flags")

class TR7AE_PT_ModelDebugProperties(bpy.types.Panel):
    bl_label = "Model Debug Info"
    bl_idname = "TR7AE_PT_ModelDebugProperties"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'TRLAU'
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        obj = context.object
        return (
            obj and
            obj.type == 'ARMATURE' and
            context.mode != 'POSE'
        )

    def draw(self, context):
        layout = self.layout
        obj = context.object

        layout.prop(obj, "max_rad")
        layout.prop(obj, "cdcRenderDataID")


        if "bone_mirror_data" in obj:
            row = layout.row()
            row.prop(obj, "tr7ae_show_mirror_data", text="", icon="TRIA_DOWN" if obj.tr7ae_show_mirror_data else "TRIA_RIGHT", emboss=False)
            row.label(text="Bone Mirror Data")

            if obj.tr7ae_show_mirror_data:
                for data in obj["bone_mirror_data"]:
                    box = layout.box()
                    box.label(text=f"Bone1: {data['bone1']} | Bone2: {data['bone2']} | Count: {data['count']}")

class ClothPointData(bpy.types.PropertyGroup):
    segment: bpy.props.IntProperty(name="Segment")
    flags: bpy.props.IntProperty(name="Flags")
    joint_order: bpy.props.IntProperty(name="Joint Order")
    up_to: bpy.props.IntProperty(name="Up To")
    pos: bpy.props.FloatVectorProperty(name="Position", size=3)

class TR7AE_PT_ClothPointInspector(bpy.types.Panel):
    bl_label = "ClothPoint"
    bl_idname = "TR7AE_PT_clothpoint_inspector"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'TRLAU'

    @classmethod
    def poll(cls, context):
        obj = context.object
        is_mesh = obj and obj.type == 'MESH'
        has_data = hasattr(obj, "cloth_points") and len(obj.cloth_points) > 0
        in_edit = context.mode == 'EDIT_MESH'
        is_vert_mode = context.tool_settings.mesh_select_mode[0]

        return is_mesh and has_data and in_edit and is_vert_mode

    def draw(self, context):
        layout = self.layout
        obj = context.object

        if context.mode != 'EDIT_MESH':
            layout.label(text="Enter Edit Mode to inspect vertices.")
            return

        bm = bmesh.from_edit_mesh(obj.data)
        selected_verts = [v for v in bm.verts if v.select]

        if not selected_verts:
            layout.label(text="Select a vertex.")
            return

        for v in selected_verts:
            idx = v.index
            if idx >= len(obj.cloth_points):
                layout.label(text=f"Vertex {idx} out of range.")
                continue

            pt = obj.cloth_points[idx]
            box = layout.box()
            box.label(text=f"Vertex {idx}")
            box.prop(pt, "segment")
            box.prop(pt, "flags")
            box.prop(pt, "joint_order")
            box.prop(pt, "up_to")
            box.prop(pt, "pos")

class TR7AE_PT_ClothJointMapInspector(bpy.types.Panel):
    bl_label = "ClothJointMap"
    bl_idname = "TR7AE_PT_clothjointmap_inspector"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'TRLAU'

    @classmethod
    def poll(cls, context):
        obj = context.object
        return (
            obj and obj.type == 'MESH'
            and obj.name == "ClothStrip"
            and hasattr(obj, "tr7ae_jointmaps")
            and context.mode == 'EDIT_MESH'
            and context.tool_settings.mesh_select_mode[0]  # Vertex select mode
        )

    def draw(self, context):
        import bmesh
        layout = self.layout
        obj = context.object

        bm = bmesh.from_edit_mesh(obj.data)
        selected_verts = {v.index for v in bm.verts if v.select}

        if not selected_verts:
            layout.label(text="Select a vertex.")
            return

        found = False
        for idx in selected_verts:
            if idx >= len(obj.cloth_points):
                continue
            current_segment = obj.cloth_points[idx].segment

            for jm in obj.tr7ae_jointmaps:
                if jm.segment != current_segment:
                    continue
                if idx not in jm.points:
                    continue

                box = layout.box()
                box.label(text=f"JointMap Segment {jm.segment} (Points: {list(jm.points)})")
                box.prop(jm, "segment")
                box.prop(jm, "flags")
                box.prop(jm, "axis")
                box.prop(jm, "joint_order")
                box.prop(jm, "center")
                box.prop(jm, "points")
                box.prop(jm, "bounds")
                found = True
                break  # Only show the first matching map

        if not found:
            layout.label(text="No joint maps use this vertex.")

class DistRuleData(bpy.types.PropertyGroup):
    point0: bpy.props.IntProperty(name="Point A")
    point1: bpy.props.IntProperty(name="Point B")
    flag0: bpy.props.IntProperty(name="Flag A")
    flag1: bpy.props.IntProperty(name="Flag B")
    min: bpy.props.FloatProperty(name="Min Distance")
    max: bpy.props.FloatProperty(name="Max Distance")

class TR7AE_PT_DistRuleInspector(bpy.types.Panel):
    bl_label = "DistRule"
    bl_idname = "TR7AE_PT_distrule_inspector"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'TRLAU'

    @classmethod
    def poll(cls, context):
        obj = context.object
        return (
            obj and obj.type == 'MESH'
            and obj.name == "ClothStrip"
            and hasattr(obj, "tr7ae_distrules")
            and context.mode == 'EDIT_MESH'
            and context.tool_settings.mesh_select_mode[1]  # Edge select mode
        )

    def draw(self, context):
        import bmesh
        layout = self.layout
        obj = context.object
        bm = bmesh.from_edit_mesh(obj.data)
        selected_edges = [e for e in bm.edges if e.select]

        if not selected_edges:
            layout.label(text="Select one or more edges.")
            return

        for e in selected_edges:
            v1 = e.verts[0].index
            v2 = e.verts[1].index

            rule = next(
                (r for r in obj.tr7ae_distrules
                 if (r.point0 == v1 and r.point1 == v2) or (r.point0 == v2 and r.point1 == v1)),
                None
            )

            box = layout.box()
            box.label(text=f"Edge ({v1}, {v2})")

            if rule:
                box.prop(rule, "flag0", text="Flag A")
                box.prop(rule, "flag1", text="Flag B")
                box.prop(rule, "min")
                box.prop(rule, "max")
            else:
                box.label(text="No DistRule found.")



class TR7AE_PT_DrawGroupPanel(Panel):
    bl_label = "Mesh"
    bl_idname = "TR7AE_PT_DrawGroupPanel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'TRLAU'

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return (
            obj is not None and
            obj.type == 'MESH' and
            not obj.name.startswith(("HSphere_", "HBox_", "HMarker_", "HCapsule", "ClothStrip", "ClothCollision_")) and
            obj.mode != 'POSE' and
            obj.type == 'MESH' and obj.get("tr7ae_type") != "ModelTarget"
        )

    def draw(self, context):
        layout = self.layout
        mesh = context.object.data
        layout.prop(mesh, "tr7ae_draw_group")
        layout.prop(mesh, "tr7ae_is_envmapped")
        layout.prop(mesh, "tr7ae_is_eyerefenvmapped")

class TR7AE_PT_SphereInfo(Panel):
    bl_label = "HSphere"
    bl_idname = "TR7AE_PT_SphereInfo"
    bl_parent_id = "TR7AE_PT_Tools"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'TRLAU'
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        bone = context.active_pose_bone
        return bone and hasattr(bone, "tr7ae_hspheres") and len(bone.tr7ae_hspheres) > 0

    def draw(self, context):
        layout = self.layout
        bone = context.active_pose_bone
        expand_state = TR7AE_PT_Tools._sphere_expand

        for i, sphere in enumerate(bone.tr7ae_hspheres):
            key = f"{bone.name}_{i}"
            expand_state.setdefault(key, False)

            row = layout.row()
            row.label(text=f"Sphere {i} (Flags: {sphere.flags})")
            op = row.operator("tr7ae.toggle_sphere", text="", icon='TRIA_DOWN' if expand_state[key] else 'TRIA_RIGHT', emboss=False)
            op.sphere_key = key

            if expand_state[key]:
                box = layout.box()
                box.label(text=f"HSphere {i}")
                box.prop(sphere, "flags")
                box.prop(sphere, "id")
                box.prop(sphere, "rank")
                box.prop(sphere, "radius")
                box.prop(sphere, "x")
                box.prop(sphere, "y")
                box.prop(sphere, "z")
                box.prop(sphere, "radius_sq")
                box.prop(sphere, "mass")
                box.prop(sphere, "buoyancy_factor")
                box.prop(sphere, "explosion_factor")
                box.prop(sphere, "material_type")
                box.prop(sphere, "pad")
                box.prop(sphere, "damage")

class TR7AE_PT_HSphereMeshInfo(bpy.types.Panel):
    bl_label = "HSphere"
    bl_idname = "TR7AE_PT_HSphereMeshInfo"
    bl_parent_id = "TR7AE_PT_Tools"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'TRLAU'
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.type == 'MESH' and obj.get("tr7ae_type") == "HSphere"

    def draw(self, context):
        layout = self.layout
        obj = context.active_object

        bone_index = obj.get("tr7ae_bone_index")
        if bone_index is None:
            layout.label(text="Missing tr7ae_bone_index")
            return

        arm = obj.find_armature()
        if not arm:
            layout.label(text="No armature found.")
            return
        bone = arm.pose.bones.get(f"Bone_{bone_index}")
        if not bone or not bone.tr7ae_hspheres:
            layout.label(text=f"No HSpheres for Bone_{bone_index}")
            return

        try:
            global_index = int(obj.name.split("_")[-1])
        except:
            layout.label(text="Invalid object name.")
            return

        current_index = 0
        target = None
        for pbone in arm.pose.bones:
            for s in pbone.tr7ae_hspheres:
                if current_index == global_index:
                    target = s
                    break
                current_index += 1
            if target:
                break

        if not target:
            layout.label(text="HSphere not found.")
            return

        layout.prop(target, "flags")
        layout.prop(target, "id")
        layout.prop(target, "rank")
        layout.prop(target, "radius")
        layout.prop(target, "x", text="X")
        layout.prop(target, "y", text="Y")
        layout.prop(target, "z", text="Z")
        layout.prop(target, "radius_sq")
        layout.prop(target, "mass")
        layout.prop(target, "buoyancy_factor")
        layout.prop(target, "explosion_factor")
        layout.prop(target, "material_type")
        layout.prop(target, "pad")
        layout.prop(target, "damage")

class TR7AE_OT_AddHSphere(bpy.types.Operator):
    bl_idname = "tr7ae.add_hsphere"
    bl_label = "Add HSphere to Bone"

    def execute(self, context):
        bone = context.active_pose_bone
        arm = context.object

        if not bone or not arm or arm.type != 'ARMATURE':
            self.report({'WARNING'}, "Select a bone in pose mode")
            return {'CANCELLED'}

        sphere = bone.tr7ae_hspheres.add()
        sphere.radius = 10.0
        sphere.x = 0.0
        sphere.y = 0.0
        sphere.z = 0.0
        sphere.id = 0
        sphere.rank = len(bone.tr7ae_hspheres) - 1

        existing_indices = [
            int(obj.name.split("_")[1])
            for obj in bpy.data.objects
            if obj.name.startswith("HSphere_") and obj.name.split("_")[1].isdigit()
        ]
        next_index = max(existing_indices, default=-1) + 1

        mesh = bpy.data.meshes.new(f"HSphereMesh_{next_index}")
        sphere_obj = bpy.data.objects.new(f"HSphere_{next_index}", mesh)
        context.collection.objects.link(sphere_obj)

        sphere_obj.display_type = 'WIRE'
        sphere_obj.show_in_front = True

        bm = bmesh.new()
        bmesh.ops.create_uvsphere(bm, u_segments=16, v_segments=8, radius=1.0)
        bm.to_mesh(mesh)
        bm.free()

        bone_mat = arm.matrix_world @ arm.data.bones[bone.name].matrix_local
        sphere_obj.location = bone_mat.to_translation() * 100
        sphere_obj.scale = (sphere.radius, sphere.radius, sphere.radius)

        sphere_obj["tr7ae_type"] = "HSphere"
        sphere_obj["tr7ae_bone_index"] = list(arm.pose.bones).index(bone)

        hinfo_empty = bpy.data.objects.get("HInfo")
        if hinfo_empty:
            sphere_obj.parent = hinfo_empty

        mod = sphere_obj.modifiers.new(name="Armature", type='ARMATURE')
        mod.object = arm

        vg = sphere_obj.vertex_groups.new(name=bone.name)
        if sphere_obj.data and sphere_obj.data.vertices:
            verts = [v.index for v in sphere_obj.data.vertices]
            vg.add(verts, 1.0, 'ADD')

        self.report({'INFO'}, f"Created HSphere_{next_index}")
        return {'FINISHED'}
    
import bpy
import bmesh
from mathutils import Vector, Quaternion, Matrix

class TR7AE_OT_AddHMarker(bpy.types.Operator):
    bl_idname = "tr7ae.add_hmarker"
    bl_label = "Add HMarker to Bone"

    def execute(self, context):
        bone = context.active_pose_bone
        arm = context.object

        if not bone or not arm or arm.type != 'ARMATURE':
            self.report({'WARNING'}, "Select a bone in pose mode")
            return {'CANCELLED'}

        bone_index = list(arm.pose.bones).index(bone)

        marker = bone.tr7ae_hmarkers.add()
        marker.px = 0.0
        marker.py = 0.0
        marker.pz = 0.0
        marker.rx = 0.0
        marker.ry = 0.0
        marker.rz = 0.0
        marker.id = 0
        marker.rank = len(bone.tr7ae_hmarkers) - 1
        marker.bone = bone_index

        existing_indices = [
            m.index
            for pbone in arm.pose.bones
            if hasattr(pbone, "tr7ae_hmarkers")
            for m in pbone.tr7ae_hmarkers
            if hasattr(m, "index")
        ]
        marker.index = max(existing_indices, default=-1) + 1

        existing_obj_indices = [
            int(obj.name.split("_")[1])
            for obj in bpy.data.objects
            if obj.name.startswith("HMarker_") and obj.name.split("_")[1].isdigit()
        ]
        next_obj_index = max(existing_obj_indices, default=-1) + 1
        marker_name = f"HMarker_{next_obj_index}"

        mesh = bpy.data.meshes.new(f"HMarkerMesh_{next_obj_index}")
        marker_obj = bpy.data.objects.new(marker_name, mesh)
        context.collection.objects.link(marker_obj)

        bm = bmesh.new()
        bmesh.ops.create_cone(
            bm,
            cap_ends=True,
            segments=8,
            radius1=2,
            radius2=0,
            depth=5
        )
        bm.to_mesh(mesh)
        bm.free()

        bone_mat = arm.matrix_world @ arm.data.bones[bone.name].matrix_local
        marker_obj.location = bone_mat.to_translation() * 100  # If using cm scale

        marker_obj["tr7ae_type"] = "HMarker"
        marker_obj["tr7ae_bone_index"] = bone_index

        hinfo_empty = bpy.data.objects.get("HInfo")
        if hinfo_empty:
            marker_obj.parent = hinfo_empty

        mod = marker_obj.modifiers.new(name="Armature", type='ARMATURE')
        mod.object = arm

        vg = marker_obj.vertex_groups.new(name=bone.name)
        if marker_obj.data and marker_obj.data.vertices:
            verts = [v.index for v in marker_obj.data.vertices]
            vg.add(verts, 1.0, 'ADD')

        self.report({'INFO'}, f"Created HMarker_{next_obj_index}")
        return {'FINISHED'}

class TR7AE_OT_AddHBox(bpy.types.Operator):
    bl_idname = "tr7ae.add_hbox"
    bl_label = "Add HBox to Bone"

    def execute(self, context):
        bone = context.active_pose_bone
        arm = context.object

        if not bone or not arm or arm.type != 'ARMATURE':
            self.report({'WARNING'}, "Select a bone in pose mode")
            return {'CANCELLED'}

        hbox = bone.tr7ae_hboxes.add()
        hbox.widthx = 5.0
        hbox.widthy = 5.0
        hbox.widthz = 5.0
        hbox.widthw = 0.0
        hbox.positionboxx = 0.0
        hbox.positionboxy = 0.0
        hbox.positionboxz = 0.0
        hbox.positionboxw = 0.0
        hbox.quatx = 0.0
        hbox.quaty = 0.0
        hbox.quatz = 0.0
        hbox.quatw = 1.0
        hbox.flags = 0
        hbox.id = 0
        hbox.rank = len(bone.tr7ae_hboxes) - 1

        existing_indices = [
            int(obj.name.split("_")[1])
            for obj in bpy.data.objects
            if obj.name.startswith("HBox_") and obj.name.split("_")[1].isdigit()
        ]
        next_index = max(existing_indices, default=-1) + 1

        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 0))
        box_obj = bpy.context.active_object
        box_obj.name = f"HBox_{next_index}"
        box_obj.display_type = 'WIRE'
        box_obj.show_in_front = True

        bone_mat = arm.matrix_world @ arm.data.bones[bone.name].matrix_local
        box_obj.location = bone_mat.to_translation() * 100

        box_obj.scale = (hbox.widthx, hbox.widthy, hbox.widthz)

        box_obj["tr7ae_type"] = "HBox"
        box_obj["tr7ae_bone_index"] = list(arm.pose.bones).index(bone)

        hinfo_empty = bpy.data.objects.get("HInfo")
        if hinfo_empty:
            box_obj.parent = hinfo_empty
            box_obj.matrix_parent_inverse.identity()

        mod = box_obj.modifiers.new(name="Armature", type='ARMATURE')
        mod.object = arm

        vg = box_obj.vertex_groups.new(name=bone.name)
        if box_obj.data and box_obj.data.vertices:
            verts = [v.index for v in box_obj.data.vertices]
            vg.add(verts, 1.0, 'ADD')

        self.report({'INFO'}, f"Created HBox_{next_index}")
        return {'FINISHED'}

class TR7AE_OT_AddHCapsule(bpy.types.Operator):
    bl_idname = "tr7ae.add_hcapsule"
    bl_label = "Add HCapsule to Bone"

    def execute(self, context):
        bone = context.active_pose_bone
        arm = context.object

        if not bone or not arm or arm.type != 'ARMATURE':
            self.report({'WARNING'}, "Select a bone in pose mode")
            return {'CANCELLED'}

        hcap = bone.tr7ae_hcapsules.add()
        hcap.radius = 2.0
        hcap.ymin = -5.0
        hcap.ymax = 5.0
        hcap.id = 0
        hcap.rank = len(bone.tr7ae_hcapsules) - 1

        existing_indices = [
            int(obj.name.split("_")[1])
            for obj in bpy.data.objects
            if obj.name.startswith("HCapsule_") and obj.name.split("_")[1].isdigit()
        ]
        next_index = max(existing_indices, default=-1) + 1
        name = f"HCapsule_{next_index}"

        mesh = bpy.data.meshes.new(f"HCapsuleMesh_{next_index}")
        obj = bpy.data.objects.new(name, mesh)
        context.collection.objects.link(obj)

        bm = bmesh.new()

        bmesh.ops.create_cone(
            bm,
            cap_ends=True,
            segments=16,
            radius1=hcap.radius,
            radius2=hcap.radius,
            depth=hcap.ymax - hcap.ymin
        )

        bm.to_mesh(mesh)
        bm.free()

        bone_mat = arm.matrix_world @ arm.data.bones[bone.name].matrix_local
        obj.location = bone_mat.to_translation() * 100

        obj.display_type = 'WIRE'
        obj.show_in_front = True
        obj.name = name

        obj["tr7ae_type"] = "HCapsule"
        obj["tr7ae_bone_index"] = list(arm.pose.bones).index(bone)

        hinfo_empty = bpy.data.objects.get("HInfo")
        if hinfo_empty:
            obj.parent = hinfo_empty
            obj.matrix_parent_inverse.identity()

        mod = obj.modifiers.new(name="Armature", type='ARMATURE')
        mod.object = arm

        vg = obj.vertex_groups.new(name=bone.name)
        if obj.data and obj.data.vertices:
            verts = [v.index for v in obj.data.vertices]
            vg.add(verts, 1.0, 'ADD')

        self.report({'INFO'}, f"Created HCapsule_{next_index}")
        return {'FINISHED'}

import bpy

class TR7AE_OT_ToggleBox(bpy.types.Operator):
    bl_idname = "tr7ae.toggle_box"
    bl_label = "Toggle HBox Section"

    box_key: bpy.props.StringProperty()

    def execute(self, context):
        expand = TR7AE_PT_Tools._box_expand
        expand[self.box_key] = not expand.get(self.box_key, False)
        return {'FINISHED'}


import bpy
from bpy.types import Panel

class TR7AE_PT_HBoxInfo(Panel):
    bl_label = "HBox"
    bl_idname = "TR7AE_PT_HBoxInfo"
    bl_parent_id = "TR7AE_PT_Tools"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'TRLAU'
    bl_options = {'DEFAULT_CLOSED'}

    _hbox_expand = {}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        if obj and obj.type == 'ARMATURE' and context.active_pose_bone:
            return bool(context.active_pose_bone.tr7ae_hboxes)
        if obj and obj.type == 'MESH' and obj.get("tr7ae_type") == "HBox":
            return True
        return False

    def draw(self, context):
        layout = self.layout
        obj = context.active_object

        if obj.type == 'MESH' and obj.get("tr7ae_type") == "HBox":
            try:
                global_index = int(obj.name.split("_")[-1])
            except:
                layout.label(text="Invalid object name.")
                return

            arm = obj.find_armature()
            if not arm:
                layout.label(text="No armature found.")
                return

            current_index = 0
            targetbox = None
            for pbone in arm.pose.bones:
                for h in pbone.tr7ae_hboxes:
                    if current_index == global_index:
                        targetbox = h
                        break
                    current_index += 1
                if targetbox:
                    break

            if not targetbox:
                layout.label(text="HBox not found.")
                return

            sub = layout.box()
            sub.label(text=f"HBox {global_index}")
            sub.prop(targetbox, "widthx")
            sub.prop(targetbox, "widthy")
            sub.prop(targetbox, "widthz")
            sub.prop(targetbox, "widthw")
            sub.prop(targetbox, "positionboxx")
            sub.prop(targetbox, "positionboxy")
            sub.prop(targetbox, "positionboxz")
            sub.prop(targetbox, "positionboxw")
            sub.prop(targetbox, "quatx")
            sub.prop(targetbox, "quaty")
            sub.prop(targetbox, "quatz")
            sub.prop(targetbox, "quatw")
            sub.prop(targetbox, "flags")
            sub.prop(targetbox, "id")
            sub.prop(targetbox, "rank")
            sub.prop(targetbox, "mass")
            sub.prop(targetbox, "buoyancy_factor")
            sub.prop(targetbox, "explosion_factor")
            sub.prop(targetbox, "material_type")
            sub.prop(targetbox, "pad")
            sub.prop(targetbox, "damage")
            sub.prop(targetbox, "pad1")
            return

        if obj.type == 'ARMATURE' and context.active_pose_bone:
            bone = context.active_pose_bone
            for i, box in enumerate(bone.tr7ae_hboxes):
                key = f"{bone.name}_{i}"
                expand = TR7AE_PT_HBoxInfo._hbox_expand.get(key, False)

                row = layout.row(align=True)
                split = row.split(factor=0.9)
                split.label(text=f"HBox {i} (Flags: {box.flags})")
                split.operator("tr7ae.toggle_hbox", text="▼" if expand else "►", emboss=False).hbox_key = key

                if expand:
                    sub = layout.box()
                    sub.prop(box, "widthx")
                    sub.prop(box, "widthy")
                    sub.prop(box, "widthz")
                    sub.prop(box, "widthw")
                    sub.prop(box, "positionboxx")
                    sub.prop(box, "positionboxy")
                    sub.prop(box, "positionboxz")
                    sub.prop(box, "positionboxw")
                    sub.prop(box, "quatx")
                    sub.prop(box, "quaty")
                    sub.prop(box, "quatz")
                    sub.prop(box, "quatw")
                    sub.prop(box, "flags")
                    sub.prop(box, "rank")
                    sub.prop(box, "mass")
                    sub.prop(box, "buoyancy_factor")
                    sub.prop(box, "explosion_factor")
                    sub.prop(box, "material_type")
                    sub.prop(box, "pad")
                    sub.prop(box, "damage")
                    sub.prop(box, "pad1")

            return

        layout.label(text="No HBox data available.")

class TR7AE_PT_AddHInfoButtons(bpy.types.Panel):
    bl_label = "Add HInfo Component"
    bl_idname = "TR7AE_PT_AddHInfoButtons"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'TRLAU'
    bl_parent_id = "TR7AE_PT_Tools"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        return context.active_pose_bone is not None

    def draw(self, context):
        layout = self.layout
        layout.label(text="Add to Active Bone:")
        layout.operator("tr7ae.add_hsphere", text="HSphere")
        layout.operator("tr7ae.add_hmarker", text="HMarker")
        layout.operator("tr7ae.add_hbox", text="HBox")
        layout.operator("tr7ae.add_hcapsule", text="HCapsule")


class TR7AE_OT_ToggleCapsule(bpy.types.Operator):
    bl_idname = "tr7ae.toggle_capsule"
    bl_label = "Toggle HCapsule Section"
    capsule_key: bpy.props.StringProperty()
    def execute(self, context):
        expand = TR7AE_PT_Tools._capsule_expand
        expand[self.capsule_key] = not expand.get(self.capsule_key, False)
        return {'FINISHED'}

class TR7AE_PT_HCapsuleInfo(Panel):
    bl_label = "HCapsule"
    bl_parent_id = "TR7AE_PT_Tools"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'TRLAU'
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        bone = context.active_pose_bone
        return bone and len(bone.tr7ae_hcapsules) > 0

    def draw(self, context):
        layout = self.layout
        bone = context.active_pose_bone
        expand = TR7AE_PT_Tools._capsule_expand

        for i, cap in enumerate(bone.tr7ae_hcapsules):
            key = f"{bone.name}_{i}"
            expand.setdefault(key, False)

            row = layout.row()
            row.label(text=f"Capsule {i} (Flags: {cap.flags})")
            op = row.operator("tr7ae.toggle_capsule", text="", 
                              icon='TRIA_DOWN' if expand[key] else 'TRIA_RIGHT', emboss=False)
            op.capsule_key = key

            if expand[key]:
                box = layout.box()
                box.label(text=f"HCapsule {i}")
                box.prop(cap, "posx"); box.prop(cap, "posy"); box.prop(cap, "posz"); box.prop(cap, "posw")
                box.prop(cap, "quatx"); box.prop(cap, "quaty"); box.prop(cap, "quatz"); box.prop(cap, "quatw")
                box.prop(cap, "flags"); box.prop(cap, "id"); box.prop(cap, "rank")
                box.prop(cap, "radius"); box.prop(cap, "length"); box.prop(cap, "mass")
                box.prop(cap, "buoyancy_factor"); box.prop(cap, "explosion_factor")
                box.prop(cap, "material_type"); box.prop(cap, "pad"); box.prop(cap, "damage")

class TR7AE_PT_HCapsuleMeshInfo(bpy.types.Panel):
    bl_label = "HCapsule"
    bl_idname = "TR7AE_PT_HCapsuleMeshInfo"
    bl_parent_id = "TR7AE_PT_Tools"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'TRLAU'
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.type == 'MESH' and obj.get("tr7ae_type") == "HCapsule"

    def draw(self, context):
        layout = self.layout
        obj = context.active_object

        bone_index = obj.get("tr7ae_bone_index")
        if bone_index is None:
            layout.label(text="Missing tr7ae_bone_index")
            return

        arm = obj.find_armature()
        if not arm:
            layout.label(text="No armature found.")
            return
        bone = arm.pose.bones.get(f"Bone_{bone_index}")
        if not bone or not bone.tr7ae_hcapsules:
            layout.label(text=f"No HCapsules for Bone_{bone_index}")
            return

        try:
            cap_idx = int(obj.name.split("_")[-1])
        except:
            cap_idx = 0
        cap = bone.tr7ae_hcapsules[cap_idx]

        layout.prop(cap, "posx");  layout.prop(cap, "posy");  layout.prop(cap, "posz");  layout.prop(cap, "posw")
        layout.prop(cap, "quatx"); layout.prop(cap, "quaty"); layout.prop(cap, "quatz"); layout.prop(cap, "quatw")
        layout.prop(cap, "flags");  layout.prop(cap, "id");    layout.prop(cap, "rank")
        layout.prop(cap, "radius"); layout.prop(cap, "length");layout.prop(cap, "mass")
        layout.prop(cap, "buoyancy_factor"); layout.prop(cap, "explosion_factor")
        layout.prop(cap, "material_type");  layout.prop(cap, "pad");  layout.prop(cap, "damage")

class TR7AE_PT_MarkersInfo(bpy.types.Panel):
    bl_label = "HMarkers"
    bl_idname = "TR7AE_PT_MarkersInfo"
    bl_parent_id = "TR7AE_PT_Tools"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'TRLAU'
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        bone = context.active_pose_bone
        return bone and hasattr(bone, "tr7ae_hmarkers") and len(bone.tr7ae_hmarkers) > 0

    def draw(self, context):
        layout = self.layout
        bone = context.active_pose_bone
        expand_state = TR7AE_PT_Tools._marker_expand

        for i, marker in enumerate(bone.tr7ae_hmarkers):
            key = f"{bone.name}_{i}"
            expand_state.setdefault(key, False)

            row = layout.row()
            row.label(text=f"Marker {i} (Index: {marker.index})")
            op = row.operator("tr7ae.toggle_marker", text="", icon='TRIA_DOWN' if expand_state[key] else 'TRIA_RIGHT', emboss=False)
            op.marker_key = key

            if expand_state[key]:
                box = layout.box()
                box.prop(marker, "bone")
                box.prop(marker, "index")
                box.label(text="Position:")
                box.prop(marker, "px", text="X")
                box.prop(marker, "py", text="Y")
                box.prop(marker, "pz", text="Z")
                box.label(text="Rotation:")
                box.prop(marker, "rx", text="X")
                box.prop(marker, "ry", text="Y")
                box.prop(marker, "rz", text="Z")

class TR7AE_PT_HMarkerMeshInfo(bpy.types.Panel):
    bl_label = "HMarker"
    bl_idname = "TR7AE_PT_HMarkerMeshInfo"
    bl_parent_id = "TR7AE_PT_Tools"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'TRLAU'
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return (
            obj and 
            obj.type == 'MESH' and 
            "HMarker" in obj.name
        )


    def draw(self, context):
        layout = self.layout
        obj = context.active_object
        layout.separator()
        layout.operator("tr7ae.snap_bone_to_marker", icon="CONSTRAINT_BONE")

        bone_index = obj.get("tr7ae_bone_index")
        if bone_index is None:
            layout.label(text="Missing tr7ae_bone_index")
            return

        arm = obj.find_armature()
        if not arm:
            layout.label(text="No armature found.")
            return

        bone_name = f"Bone_{bone_index}"
        pbone = arm.pose.bones.get(bone_name)
        if not pbone or not hasattr(pbone, "tr7ae_hmarkers"):
            layout.label(text=f"No HMarkers for {bone_name}")
            return

        try:
            marker_index = int(obj.name.split("_")[-1])
        except:
            layout.label(text="Invalid HMarker name format.")
            return

        obj = context.object
        if not obj or not obj.name.startswith("HMarker_"):
            layout.label(text="Select an HMarker.")
            return

        try:
            marker_index = int(obj.name.split("_")[1])
        except:
            layout.label(text="Invalid HMarker name.")
            return

        armature = obj.find_armature()
        if not armature:
            layout.label(text="HMarker not parented to armature.")
            return

        global_index = 0
        marker = None
        for pbone in armature.pose.bones:
            for m in pbone.tr7ae_hmarkers:
                if global_index == marker_index:
                    marker = m
                    break
                global_index += 1
            if marker:
                break

        if not marker:
            layout.label(text="Marker not found.")
            return

        layout.prop(marker, "bone")
        layout.prop(marker, "index")
        layout.label(text="Position:")
        layout.prop(marker, "px", text="X")
        layout.prop(marker, "py", text="Y")
        layout.prop(marker, "pz", text="Z")
        layout.label(text="Rotation:")
        layout.prop(marker, "rx", text="X")
        layout.prop(marker, "ry", text="Y")
        layout.prop(marker, "rz", text="Z")

class TR7AE_OT_ToggleSphere(bpy.types.Operator):
    bl_idname = "tr7ae.toggle_sphere"
    bl_label = "Toggle HSphere Section"

    sphere_key: bpy.props.StringProperty()

    def execute(self, context):
        expand = TR7AE_PT_Tools._sphere_expand
        expand[self.sphere_key] = not expand.get(self.sphere_key, False)
        return {'FINISHED'}
    
class TR7AE_OT_ToggleHBox(bpy.types.Operator):
    bl_idname = "tr7ae.toggle_hbox"
    bl_label = "Toggle HBox Section"

    hbox_key: bpy.props.StringProperty()

    def execute(self, context):
        expand = TR7AE_PT_HBoxInfo._hbox_expand
        expand[self.hbox_key] = not expand.get(self.hbox_key, False)
        return {'FINISHED'}

class TR7AE_OT_ToggleMarker(bpy.types.Operator):
    bl_idname = "tr7ae.toggle_marker"
    bl_label = "Toggle HMarker Section"

    marker_key: bpy.props.StringProperty()

    def execute(self, context):
        expand = TR7AE_PT_Tools._marker_expand
        expand[self.marker_key] = not expand.get(self.marker_key, False)
        return {'FINISHED'}
    
class TR7AE_OT_SnapBoneToHMarker(bpy.types.Operator):
    bl_idname = "tr7ae.snap_bone_to_marker"
    bl_label = "Snap Skeleton to HMarker"
    bl_description = "Snap the first bone of the selected armature to the HMarker mesh and parent the armature to it"

    def execute(self, context):
        import mathutils

        objs = context.selected_objects
        hmarker = None
        armature = None

        for obj in objs:
            if obj.get("tr7ae_type") == "HMarker":
                hmarker = obj
            elif obj.type == 'ARMATURE':
                armature = obj

        if not hmarker or not armature:
            self.report({'ERROR'}, "Select one armature and one HMarker")
            return {'CANCELLED'}

        if not armature.pose.bones:
            self.report({'ERROR'}, "Armature has no bones")
            return {'CANCELLED'}

        first_bone = armature.pose.bones[0]

        bpy.ops.object.mode_set(mode='OBJECT')
        context.view_layer.objects.active = armature
        bpy.ops.object.mode_set(mode='POSE')

        target_matrix = hmarker.matrix_world.copy()

        relative_matrix = armature.matrix_world.inverted() @ target_matrix

        location = relative_matrix.to_translation()
        rotation = relative_matrix.to_euler()
        clean_matrix = mathutils.Euler(rotation).to_matrix().to_4x4()
        clean_matrix.translation = location

        first_bone.matrix_basis = clean_matrix

        bpy.ops.object.mode_set(mode='OBJECT')

        armature.parent = hmarker
        armature.matrix_parent_inverse = hmarker.matrix_world.inverted()

        self.report({'INFO'}, f"{first_bone.name} aligned and armature parented to {hmarker.name}")
        return {'FINISHED'}
    
class TR7AE_OT_ToggleHInfoVisibility(bpy.types.Operator):
    bl_idname = "tr7ae.toggle_hinfo_visibility"
    bl_label = "Toggle HInfo Visibility"
    bl_description = "Toggle visibility of all HInfo components (HSpheres, HBoxes, HCapsules, HMarkers)"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        hinfo_objects = [
            obj for obj in bpy.data.objects
            if obj.name.startswith(("HSphere_", "HBox_", "HCapsule_", "HMarker_"))
        ]

        if not hinfo_objects:
            self.report({'WARNING'}, "No HInfo objects found.")
            return {'CANCELLED'}

        any_visible = any(not obj.hide_viewport for obj in hinfo_objects)
        new_state = True if any_visible else False

        for obj in hinfo_objects:
            obj.hide_viewport = new_state
            obj.hide_render = new_state

        return {'FINISHED'}

class TR7AE_PT_VisibilityPanel(bpy.types.Panel):
    bl_label = "Visibility"
    bl_idname = "TR7AE_PT_visibility"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'TRLAU'
    bl_options = {'DEFAULT_CLOSED'}

    _sphere_expand = {}
    _box_expand = {}
    _marker_expand = {}

    @classmethod
    def poll(cls, context):
        return any(
            obj.name.startswith(("HSphere_", "HMarker_", "HBox_"))
            for obj in bpy.data.objects
        )

    def draw(self, context):
        layout = self.layout

        layout.operator("tr7ae.toggle_hinfo_visibility", icon="HIDE_OFF")

        hsphere_exists = any(obj.name.startswith("HSphere_") for obj in bpy.data.objects)
        if hsphere_exists:
            layout.operator("tr7ae.toggle_hspheres", icon='RESTRICT_VIEW_OFF')

        hmarker_exists = any(obj.name.startswith("HMarker_") for obj in bpy.data.objects)
        if hmarker_exists:
            layout.operator("tr7ae.toggle_hmarkers", icon='RESTRICT_VIEW_OFF')

        hbox_exists = any(obj.name.startswith("HBox_") for obj in bpy.data.objects)
        if hbox_exists:
            layout.operator("tr7ae.toggle_hboxes", icon='RESTRICT_VIEW_OFF')

        hcapsule_exists = any(obj.name.startswith("HCapsule_") for obj in bpy.data.objects)
        if hcapsule_exists:
            layout.operator("tr7ae.toggle_hcapsules", icon='RESTRICT_VIEW_OFF')

class TR7AE_PT_MaterialPanel(Panel):
    bl_label = "Material"
    bl_idname = "TR7AE_PT_MaterialPanel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'TRLAU'

    @classmethod
    def poll(cls, context):
        mat = context.object.active_material if context.object else None
        return mat and hasattr(mat, "tr7ae_texture_id")

    def draw(self, context):
        layout = self.layout
        mat = context.object.active_material

        layout.prop(mat, "tr7ae_texture_id")
        layout.prop(mat, "tr7ae_blend_value")
        layout.prop(mat, "tr7ae_unknown_1")
        layout.prop(mat, "tr7ae_unknown_2")
        layout.prop(mat, "tr7ae_single_sided")
        layout.prop(mat, "tr7ae_texture_wrap")
        layout.prop(mat, "tr7ae_unknown_3")
        layout.prop(mat, "tr7ae_unknown_4")
        layout.prop(mat, "tr7ae_flat_shading")
        layout.prop(mat, "tr7ae_sort_z")
        layout.prop(mat, "tr7ae_stencil_pass")
        layout.prop(mat, "tr7ae_stencil_func")
        layout.prop(mat, "tr7ae_alpha_ref")

class NextGenMaterialPanel(bpy.types.Panel):
    bl_label = "Next Gen Material"
    bl_idname = "TR7AE_PT_NextGenMaterialPanel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'TRLAU'

    @classmethod
    def poll(cls, context):
        obj = context.object
        return obj is not None and obj.type == 'MESH' and obj.active_material is not None

    def draw(self, context):
        layout = self.layout
        mat = context.object.active_material
        props = mat.nextgen_material_properties

        layout.prop(props, "mat_id")
        layout.prop(props, "blend_mode")
        layout.prop(props, "combiner_type")
        layout.prop(props, "flags")
        layout.prop(props, "vertex_shader_flags")
        layout.prop(props, "double_sided")
        layout.prop(props, "opacity")
        layout.prop(props, "poly_flags")
        layout.prop(props, "uv_auto_scroll_speed")
        layout.prop(props, "sort_bias")
        layout.prop(props, "detail_range_mul")
        layout.prop(props, "detail_scale")
        layout.prop(props, "parallax_scale")
        layout.prop(props, "parallax_offset")
        layout.prop(props, "specular_power")
        layout.prop(props, "specular_shift_0")
        layout.prop(props, "specular_shift_1")
        layout.prop(props, "rim_light_color")
        layout.prop(props, "rim_light_intensity")
        layout.prop(props, "water_blend_bias")
        layout.prop(props, "water_blend_exponent")
        layout.prop(props, "water_deep_color")

def on_blend_value_changed(self, context):
    mat = self
    if not mat.use_nodes:
        return

    if getattr(mat, 'tr7ae_is_envmapped', False) or getattr(mat, 'tr7ae_is_eyerefenvmapped', False):
        return

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    old_image = None
    for node in nodes:
        if node.type == 'TEX_IMAGE' and node.image:
            old_image = node.image
            break

    if old_image and "_NonColor" in old_image.name:
        base_name = old_image.name.replace("_NonColor", "")
        if base_name in bpy.data.images:
            old_image = bpy.data.images[base_name]

    nodes.clear()

    output = nodes.new(type='ShaderNodeOutputMaterial')
    output.location = (600, 0)

    bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    bsdf.location = (300, 0)
    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])

    tex_image = nodes.new(type='ShaderNodeTexImage')
    tex_image.location = (-400, 0)
    if old_image:
        tex_image.image = old_image

    attr = nodes.new(type='ShaderNodeAttribute')
    attr.attribute_name = "Color"
    attr.location = (-600, 100)

    multiply_rgb = nodes.new(type='ShaderNodeMixRGB')
    multiply_rgb.blend_type = 'MULTIPLY'
    multiply_rgb.inputs['Fac'].default_value = 1.0
    multiply_rgb.location = (-200, 0)

    links.new(attr.outputs['Color'], multiply_rgb.inputs['Color2'])
    links.new(tex_image.outputs['Color'], multiply_rgb.inputs['Color1'])
    links.new(multiply_rgb.outputs['Color'], bsdf.inputs['Base Color'])

    if old_image and "_NonColor" in old_image.name:
        base_name = old_image.name.replace("_NonColor", "")
        if base_name in bpy.data.images:
            old_image = bpy.data.images[base_name]

    if old_image and "_NonColor" in old_image.name:
        base_name = old_image.name.replace("_NonColor", "")
        if base_name in bpy.data.images:
            old_image = bpy.data.images[base_name]

    elif mat.tr7ae_blend_value == 1:
        vc_alpha = attr.outputs['Alpha']
        tex_alpha = tex_image.outputs['Alpha']

        alpha_mult = nodes.new(type='ShaderNodeMath')
        alpha_mult.operation = 'MULTIPLY'
        alpha_mult.location = (0, -200)

        links.new(tex_alpha, alpha_mult.inputs[0])
        links.new(vc_alpha, alpha_mult.inputs[1])
        bsdf.inputs['Specular IOR Level'].default_value = 0.0
        links.new(alpha_mult.outputs[0], bsdf.inputs['Alpha'])

        mat.blend_method = 'BLEND'

    elif mat.tr7ae_blend_value == 0:
        nodes.clear()

        output = nodes.new(type='ShaderNodeOutputMaterial')
        output.location = (400, 0)

        bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
        bsdf.location = (200, 0)
        bsdf.inputs['Specular IOR Level'].default_value = 0.0
        bsdf.inputs['Roughness'].default_value = 0.5

        tex_image = nodes.new(type='ShaderNodeTexImage')
        tex_image.location = (-600, 100)
        if old_image:
            tex_image.image = old_image
        tex_image.image.colorspace_settings.name = "sRGB"

        attr = nodes.new(type='ShaderNodeAttribute')
        attr.attribute_name = "Color"
        attr.location = (-600, -100)

        mult_node = nodes.new(type='ShaderNodeMixRGB')
        mult_node.blend_type = 'MULTIPLY'
        mult_node.inputs['Fac'].default_value = 1.0
        mult_node.location = (-300, 0)

        greater = nodes.new(type='ShaderNodeMath')
        greater.operation = 'GREATER_THAN'
        greater.inputs[1].default_value = 0.0
        greater.location = (-300, -200)

        links.new(tex_image.outputs['Color'], mult_node.inputs['Color1'])
        links.new(attr.outputs['Color'], mult_node.inputs['Color2'])

        links.new(tex_image.outputs['Alpha'], greater.inputs[0])

        links.new(mult_node.outputs['Color'], bsdf.inputs['Base Color'])
        links.new(greater.outputs[0], bsdf.inputs['Alpha'])
        links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])

        mat.blend_method = 'OPAQUE'


    elif mat.tr7ae_blend_value == (1, 7):
        nodes.clear()
        output = nodes.new(type='ShaderNodeOutputMaterial')
        output.location = (400, 0)

        bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
        bsdf.location = (0, 0)
        bsdf.inputs['Specular IOR Level'].default_value = 0.0

        tex_image = nodes.new(type='ShaderNodeTexImage')
        tex_image.location = (-600, 0)
        if old_image:
            tex_image.image = old_image
        tex_image.image.colorspace_settings.name = "sRGB"

        attr = nodes.new(type='ShaderNodeAttribute')
        attr.attribute_name = "Color"
        attr.location = (-600, -200)

        alpha_mult = nodes.new(type='ShaderNodeMath')
        alpha_mult.operation = 'MULTIPLY'
        alpha_mult.location = (-200, -100)

        links.new(tex_image.outputs['Alpha'], alpha_mult.inputs[0])
        links.new(attr.outputs['Alpha'], alpha_mult.inputs[1])
        links.new(alpha_mult.outputs[0], bsdf.inputs['Alpha'])
        links.new(tex_image.outputs['Color'], bsdf.inputs['Base Color'])
        links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])

        mat.blend_method = 'BLEND'

    elif mat.tr7ae_blend_value == 9:
        if old_image and "_NonColor" in old_image.name:
            base_name = old_image.name.replace("_NonColor", "")
            original = bpy.data.images.get(base_name)
            if original:
                old_image = original
        mat.use_nodes = True
        mat.blend_method = 'BLEND'

        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()

        tex_image = nodes.new(type='ShaderNodeTexImage')
        tex_image.location = (-600, 0)
        if old_image:
            tex_image.image = old_image

        multiply = nodes.new(type='ShaderNodeMath')
        multiply.operation = 'MULTIPLY'
        multiply.inputs[1].default_value = 1.0
        multiply.location = (-400, 0)

        glossy = nodes.new(type='ShaderNodeBsdfGlossy')
        glossy.inputs['Roughness'].default_value = 0.3
        glossy.location = (-200, 0)

        transparent = nodes.new(type='ShaderNodeBsdfTransparent')
        transparent.location = (-200, -200)

        mix_shader = nodes.new(type='ShaderNodeMixShader')
        mix_shader.location = (0, 0)

        mix_value = nodes.new(type='ShaderNodeValue')
        mix_value.outputs[0].default_value = 0.1
        mix_value.location = (-400, -200)

        output = nodes.new(type='ShaderNodeOutputMaterial')
        output.location = (200, 0)

        links.new(tex_image.outputs['Alpha'], multiply.inputs[0])
        links.new(multiply.outputs[0], glossy.inputs['Color'])

        links.new(glossy.outputs['BSDF'], mix_shader.inputs[2])
        links.new(transparent.outputs['BSDF'], mix_shader.inputs[1])
        links.new(mix_value.outputs[0], mix_shader.inputs['Fac'])
        links.new(mix_shader.outputs['Shader'], output.inputs['Surface'])

    elif mat.tr7ae_blend_value in (2, 5, 8):
        nodes.clear()
        if old_image:
            base_name = old_image.name
            if "_NonColor" in base_name:
                base_name = base_name.replace("_NonColor", "")
                original = bpy.data.images.get(base_name)
                if original:
                    old_image = original

            image_copy_name = old_image.name + "_NonColor"

            if image_copy_name in bpy.data.images:
                image_copy = bpy.data.images[image_copy_name]
            else:
                image_copy = old_image.copy()
                image_copy.name = image_copy_name
                image_copy.colorspace_settings.name = "Non-Color"
        else:
            image_copy = None

        output = nodes.new(type='ShaderNodeOutputMaterial')
        output.location = (400, 0)

        tex_image = nodes.new(type='ShaderNodeTexImage')
        tex_image.location = (-600, 0)
        if image_copy:
            tex_image.image = image_copy

        add_color = nodes.new(type='ShaderNodeMixRGB')
        add_color.blend_type = 'ADD'
        add_color.inputs['Fac'].default_value = 1.0
        add_color.location = (-400, 0)

        transparent = nodes.new(type='ShaderNodeBsdfTransparent')
        transparent.location = (-200, 0)

        add_shader = nodes.new(type='ShaderNodeAddShader')
        add_shader.location = (0, 0)

        links.new(tex_image.outputs['Color'], add_color.inputs['Color1'])

        links.new(add_color.outputs['Color'], transparent.inputs['Color'])

        links.new(transparent.outputs['BSDF'], add_shader.inputs[0])
        links.new(transparent.outputs['BSDF'], add_shader.inputs[1])

        links.new(add_shader.outputs['Shader'], output.inputs['Surface'])

        mat.blend_method = 'BLEND'

    elif mat.tr7ae_blend_value == 3:
        nodes.clear()
        output = nodes.new(type='ShaderNodeOutputMaterial')
        output.location = (400, 0)

        if old_image and "_NonColor" in old_image.name:
            base_name = old_image.name.replace("_NonColor", "")
            if base_name in bpy.data.images:
                old_image = bpy.data.images[base_name]

        tex_image = nodes.new(type='ShaderNodeTexImage')
        tex_image.location = (-600, 0)
        if old_image:
            tex_image.image = old_image
        tex_image.image.colorspace_settings.name = "sRGB"

        multiply = nodes.new(type='ShaderNodeMixRGB')
        multiply.blend_type = 'MULTIPLY'
        multiply.inputs['Fac'].default_value = 0.5
        multiply.inputs['Color1'].default_value = (1.0, 1.0, 1.0, 1.0)
        multiply.location = (-400, 0)
        links.new(tex_image.outputs['Color'], multiply.inputs['Color2'])

        transparent = nodes.new(type='ShaderNodeBsdfTransparent')
        transparent.location = (-200, 0)
        links.new(multiply.outputs['Color'], transparent.inputs['Color'])

        links.new(transparent.outputs['BSDF'], output.inputs['Surface'])

        mat.blend_method = 'BLEND'

    elif mat.tr7ae_blend_value == 4:
        nodes.clear()

        output = nodes.new(type='ShaderNodeOutputMaterial')
        output.location = (400, 0)

        tex_image = nodes.new(type='ShaderNodeTexImage')
        tex_image.location = (-600, 0)
        if old_image:
            tex_image.image = old_image
        tex_image.image.colorspace_settings.name = "sRGB"

        bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
        bsdf.location = (-300, 150)
        bsdf.inputs['Alpha'].default_value = 0.0

        emission = nodes.new(type='ShaderNodeEmission')
        emission.location = (-300, -50)
        emission.inputs['Color'].default_value = (1.0, 1.0, 1.0, 1.0)
        emission.inputs['Strength'].default_value = 0.0

        mix = nodes.new(type='ShaderNodeMixShader')
        mix.location = (100, 50)
        mix.inputs['Fac'].default_value = 0.5

        links.new(bsdf.outputs['BSDF'], mix.inputs[1])
        links.new(emission.outputs['Emission'], mix.inputs[2])
        links.new(mix.outputs['Shader'], output.inputs['Surface'])

        mat.blend_method = 'BLEND'

    elif mat.tr7ae_blend_value == 6:
        nodes.clear()

        output = nodes.new(type='ShaderNodeOutputMaterial')
        output.location = (200, 0)

        tex_image = nodes.new(type='ShaderNodeTexImage')
        tex_image.location = (-400, 0)
        if old_image:
            tex_image.image = old_image
        tex_image.image.colorspace_settings.name = "sRGB"

        transparent = nodes.new(type='ShaderNodeBsdfTransparent')
        transparent.location = (0, 0)
        links.new(transparent.outputs['BSDF'], output.inputs['Surface'])

        mat.blend_method = 'BLEND'

def register_material_properties():
    bpy.types.Material.tr7ae_texture_id = IntProperty(
        name="Texture ID",
        description="ID of the texture to apply",
        min=0,
        max=8191,
        default=0
    )
    bpy.types.Material.tr7ae_blend_value = IntProperty(
        name="Blend Value",
        description="Blending mode\n\n0 = Opaque\n1 = Alpha Blend\n2 = Additive\n3 = Destination Alpha\n4 = Destination Add\n5 = Blend 5050\n6 = Invisible?\n7 = Multipass Alpha\n8 = Destination Alpha Source Only",
        min=0,
        max=15,
        default=0,
        update=on_blend_value_changed
    )
    bpy.types.Material.tr7ae_unknown_1 = IntProperty(
        name="Unknown 1",
        description="Unknown 1",
        min=0,
        max=7,
        default=0
    )
    bpy.types.Material.tr7ae_unknown_2 = bpy.props.BoolProperty(
        name="Unknown 2",
        description="Unknown 2",
        default=0
    )
    def get_single_sided(self):
        return self.use_backface_culling

    def set_single_sided(self, value):
        self.use_backface_culling = value

    bpy.types.Material.tr7ae_single_sided = bpy.props.BoolProperty(
        name="Single Sided",
        description="Whether the material is single sided (enables backface culling)",
        get=get_single_sided,
        set=set_single_sided
    )
    bpy.types.Material.tr7ae_texture_wrap = IntProperty(
        name="Texture Wrap",
        description="Texture Wrapping",
        min=0,
        max=3,
        default=0
    )
    bpy.types.Material.tr7ae_unknown_3 = bpy.props.BoolProperty(
        name="Unknown 3",
        description="Unknown 3",
        default=0
    )
    bpy.types.Material.tr7ae_unknown_4 = bpy.props.BoolProperty(
        name="Unknown 4",
        description="Unknown 4",
        default=0
    )
    bpy.types.Material.tr7ae_flat_shading = bpy.props.BoolProperty(
        name="Flat Shading",
        description="Stops the mesh from reacting to light in any way",
        default=0
    )
    bpy.types.Material.tr7ae_sort_z = bpy.props.BoolProperty(
        name="Sort Z",
        description="Sort Z",
        default=0
    )
    bpy.types.Material.tr7ae_stencil_pass = IntProperty(
        name="Stencil Pass",
        description="Stencil Pass",
        min=0,
        max=3,
        default=0
    )
    bpy.types.Material.tr7ae_stencil_func = bpy.props.BoolProperty(
        name="Stencil Func",
        description="Stencil Func",
        default=0
    )
    bpy.types.Material.tr7ae_alpha_ref = bpy.props.BoolProperty(
        name="Alpha Ref",
        description="Alpha Ref",
        default=0
    )
    bpy.types.Material.tr7ae_is_envmapped = bpy.props.BoolProperty(
        name="Environment Mapping",
        description="Whether this material uses environment mapping",
        default=False
    )
    bpy.types.Material.tr7ae_is_eyerefenvmapped = bpy.props.BoolProperty(
        name="Eye Reflection Environment Mapping",
        description="Whether this material uses eye reflection environment mapping",
        default=False
    )

def register_panel_properties():
    bpy.types.WindowManager.tr7ae_expand_sphere = bpy.props.BoolProperty(default=True)
    bpy.types.WindowManager.tr7ae_expand_box = bpy.props.BoolProperty(default=True)
    bpy.types.WindowManager.tr7ae_expand_markers = bpy.props.BoolProperty(default=True)

def unregister_panel_properties():
    del bpy.types.WindowManager.tr7ae_expand_sphere
    del bpy.types.WindowManager.tr7ae_expand_box
    del bpy.types.WindowManager.tr7ae_expand_markers

def unregister_material_properties():
    del bpy.types.Material.tr7ae_texture_id
    del bpy.types.Material.tr7ae_blend_value
    del bpy.types.Material.tr7ae_unknown_1
    del bpy.types.Material.tr7ae_unknown_2
    del bpy.types.Material.tr7ae_single_sided
    del bpy.types.Material.tr7ae_texture_wrap
    del bpy.types.Material.tr7ae_unknown_3
    del bpy.types.Material.tr7ae_unknown_4
    del bpy.types.Material.tr7ae_flat_shading
    del bpy.types.Material.tr7ae_sort_z
    del bpy.types.Material.tr7ae_stencil_pass
    del bpy.types.Material.tr7ae_stencil_func
    del bpy.types.Material.tr7ae_alpha_ref

def register_envmap_property():
    bpy.types.Mesh.tr7ae_is_envmapped = bpy.props.BoolProperty(
        name="Environment Mapping",
        description="Whether this mesh uses environment reflection",
        default=False
    )
def register_eyerefenvmap_property():
    bpy.types.Mesh.tr7ae_is_eyerefenvmapped = bpy.props.BoolProperty(
        name="Eye Reflection Environment Mapping",
        description="Whether this mesh uses eye reflection environment mapping",
        default=False
    )

def unregister_envmap_property():
    del bpy.types.Mesh.tr7ae_is_envmapped

def unregister_eyerefenvmap_property():
    del bpy.types.Mesh.tr7ae_is_eyerefenvmapped


def register_draw_group_property():
    bpy.types.Mesh.tr7ae_draw_group = IntProperty(
        name="Draw Group",
        description="Declares wether the mesh only renders under certain situations.\n0 = Mesh is always rendered",
        default=0
    )

def unregister_draw_group_property():
    del bpy.types.Mesh.tr7ae_draw_group

@persistent
def tr7ae_sync_handler(depsgraph):
    scene = bpy.context.scene
    for obj in scene.objects:
        obj_type = obj.get("tr7ae_type")
        bone_index = obj.get("tr7ae_bone_index")
        if obj_type != "HSphere" or bone_index is None:
            continue

        arm = obj.find_armature()
        if not arm:
            continue

        try:
            global_index = int(obj.name.split("_")[-1])
        except:
            continue

        current_index = 0
        target = None
        bone_name = None
        for pbone in arm.pose.bones:
            for s in pbone.tr7ae_hspheres:
                if current_index == global_index:
                    target = s
                    bone_name = pbone.name
                    break
                current_index += 1
            if target:
                break

        if not target or not bone_name:
            continue

        bone_matrix = arm.matrix_world @ arm.data.bones[bone_name].matrix_local
        local_pos = bone_matrix.inverted() @ obj.matrix_world.translation

        target.x = local_pos.x
        target.y = local_pos.y
        target.z = local_pos.z
        target.radius = obj.scale.x

@persistent
def sync_hbox_transforms(scene):
    for obj in scene.objects:
        if obj.get("tr7ae_type") != "HBox":
            continue

        arm = obj.find_armature()
        if not arm:
            continue

        try:
            global_index = int(obj.name.split("_")[-1])
        except:
            continue

        current_index = 0
        targetbox = None
        bone_name = None

        for pbone in arm.pose.bones:
            for h in pbone.tr7ae_hboxes:
                if current_index == global_index:
                    targetbox = h
                    bone_name = pbone.name
                    break
                current_index += 1
            if targetbox:
                break

        if not targetbox or not bone_name:
            continue

        bone_mat = arm.matrix_world @ arm.data.bones[bone_name].matrix_local
        local_mat = bone_mat.inverted() @ obj.matrix_world

        loc = local_mat.to_translation()
        targetbox.positionboxx = loc.x
        targetbox.positionboxy = loc.y
        targetbox.positionboxz = loc.z

        targetbox.widthx = obj.scale.x
        targetbox.widthy = obj.scale.y
        targetbox.widthz = obj.scale.z

        quat = obj.matrix_world.to_quaternion()
        targetbox.quatx = quat.x
        targetbox.quaty = quat.y
        targetbox.quatz = quat.z
        targetbox.quatw = quat.w


@persistent
def sync_hcapsule_transforms(scene):
    for obj in scene.objects:
        if obj.get("tr7ae_type") != "HCapsule":
            continue

        arm = obj.find_armature()
        if not arm:
            continue

        bone_index = obj.get("tr7ae_bone_index")
        if bone_index is None:
            continue

        bone_name = f"Bone_{bone_index}"
        pbone = arm.pose.bones.get(bone_name)
        if not pbone or not pbone.tr7ae_hcapsules:
            continue

        try:
            cap_idx = int(obj.name.split("_")[-1])
        except:
            cap_idx = 0
        cap = pbone.tr7ae_hcapsules[cap_idx]

        bone_mat  = arm.matrix_world @ arm.data.bones[bone_name].matrix_local
        local_mat = bone_mat.inverted() @ obj.matrix_world

        loc = local_mat.to_translation()
        cap.posx = loc.x
        cap.posy = loc.y
        cap.posz = loc.z

        quat = local_mat.to_quaternion()
        cap.quatx = quat.x
        cap.quaty = quat.y
        cap.quatz = quat.z
        cap.quatw = quat.w

        cap.radius = obj.scale.x
        cap.length = obj.scale.z

@persistent
def sync_hmarker_transforms(scene):
    for obj in scene.objects:
        if obj.get("tr7ae_type") != "HMarker":
            continue

        arm = obj.find_armature()
        if not arm:
            continue

        try:
            global_index = int(obj.name.split("_")[-1])
        except:
            continue

        current_index = 0
        marker = None
        bone_name = None
        for bone in arm.pose.bones:
            for m in bone.tr7ae_hmarkers:
                if current_index == global_index:
                    marker = m
                    bone_name = bone.name
                    break
                current_index += 1
            if marker:
                break

        if not marker or bone_name is None:
            continue

        world_mat = obj.matrix_world

        bone_mat = arm.matrix_world @ arm.data.bones[bone_name].matrix_local
        local_mat = bone_mat.inverted() @ world_mat

        pos = local_mat.to_translation()
        marker.px = pos.x
        marker.py = pos.y
        marker.pz = pos.z

        rot = local_mat.to_euler(obj.rotation_mode)
        marker.rx = rot.x
        marker.ry = rot.y
        marker.rz = rot.z

@persistent
def sync_model_target_properties(scene):
    for obj in scene.objects:
        if obj.get("tr7ae_type") != "ModelTarget":
            continue
        if not hasattr(obj, "tr7ae_modeltarget"):
            continue

        segment = obj.get("tr7ae_segment")
        if segment is None:
            continue

        armature = obj.find_armature()
        if not armature or armature.type != 'ARMATURE':
            continue

        bone_name = f"Bone_{segment}"
        bone = armature.pose.bones.get(bone_name)
        if not bone:
            continue

        bone_matrix_world = armature.matrix_world @ armature.data.bones[bone_name].matrix_local

        target_matrix_world = obj.matrix_world

        relative_matrix = bone_matrix_world.inverted() @ target_matrix_world

        loc = relative_matrix.to_translation()
        rot = relative_matrix.to_euler('XYZ')

        obj.tr7ae_modeltarget.px = loc.x
        obj.tr7ae_modeltarget.py = loc.y
        obj.tr7ae_modeltarget.pz = loc.z
        obj.tr7ae_modeltarget.rx = rot.x
        obj.tr7ae_modeltarget.ry = rot.y
        obj.tr7ae_modeltarget.rz = rot.z

def simulate_cloth_on_frame(scene):
    import numpy as np
    import mathutils

    obj = bpy.data.objects.get("ClothStrip")
    if not obj or "cloth_points" not in obj or not hasattr(obj, "tr7ae_distrules"):
        return

    mesh = obj.data
    if not mesh or len(obj.cloth_points) != len(mesh.vertices):
        return

    num_points = len(obj.cloth_points)
    dt = 1 / scene.render.fps
    gravity = mathutils.Vector((0, 0, -999999.0))  # faster fall
    damping = 0.1
    stiffness = 1
    iterations = 40

    if scene.frame_current == 1 or cloth_sim_state.get("positions") is None:
        positions = np.array([v.co for v in mesh.vertices])
        velocities = np.zeros_like(positions)
        cloth_sim_state["positions"] = positions
        cloth_sim_state["velocities"] = velocities
        cloth_sim_state["last_frame"] = -1
    else:
        positions = cloth_sim_state["positions"]
        velocities = cloth_sim_state["velocities"]

    if cloth_sim_state["last_frame"] == scene.frame_current:
        return

    for i, pt in enumerate(obj.cloth_points):
        if pt.flags not in (1, 5):
            velocities[i] *= damping
            velocities[i] += np.array(gravity) * dt
            positions[i] += velocities[i] * dt

    for _ in range(iterations):
        for rule in obj.tr7ae_distrules:
            i, j = rule.point0, rule.point1
            if i >= num_points or j >= num_points or i == j:
                continue

            pi = positions[i]
            pj = positions[j]
            delta = pj - pi
            dist = np.linalg.norm(delta)
            if dist == 0:
                continue

            direction = delta / dist
            rest_length = np.linalg.norm(mesh.vertices[i].co - mesh.vertices[j].co)
            diff = dist - rest_length
            correction = stiffness * 0.5 * diff * direction

            if obj.cloth_points[i].flags not in (1, 5):
                positions[i] += correction
            if obj.cloth_points[j].flags not in (1, 5):
                positions[j] -= correction

    for i, v in enumerate(mesh.vertices):
        v.co = positions[i].tolist()

    mesh.update()
    cloth_sim_state["positions"] = positions
    cloth_sim_state["velocities"] = velocities
    cloth_sim_state["last_frame"] = scene.frame_current

def read_octree_sphere_node(f, offset):
    f.seek(offset)
    raw = f.read(56)
    if len(raw) < 56:
        raise ValueError("Incomplete OctreeSpherae node (expected 56 bytes)")

    x, y, z, r, strip, num_spheres, *spheres = struct.unpack("<4f2i8i", raw)
    return {
        "boundsphere": (x, y, z, r),
        "strip": strip,
        "num_spheres": num_spheres,
        "spheres": spheres[:num_spheres]
    }


def read_strip_indices(f, offset, header_size):
    f.seek(offset + header_size)
    vertex_count = struct.unpack("<i", f.read(4))[0]
    f.read(16)
    mat_idx = struct.unpack("<i", f.read(4))[0]
    f.read(16)
    nextTexture = struct.unpack("<i", f.read(4))[0]
    strip_indices = list(struct.unpack(f"<{vertex_count}h", f.read(vertex_count * 2)))
    return strip_indices

def split_strip_on_restart_markers(strip, marker=-1):
    strips = []
    current = []
    for idx in strip:
        if idx == marker:
            if len(current) >= 3:
                strips.append(current)
            current = []
        else:
            current.append(idx)
    if len(current) >= 3:
        strips.append(current)
    return strips

def convert_indices_to_tris(index_list):
    tris = []
    for i in range(0, len(index_list), 3):
        if i + 2 >= len(index_list):
            break
        i0, i1, i2 = index_list[i], index_list[i+1], index_list[i+2]
        if i0 != i1 and i1 != i2 and i0 != i2:
            tris.append((i0, i1, i2))
    return tris


def traverse_octree(filepath, octree_ptr, section_index):
    section_path = filepath.parent / f"{section_index}_0.gnc"
    face_indices = []

    with open(section_path, "rb") as f:
        assert f.read(4) == b'SECT'
        f.seek(0xC)
        packed = struct.unpack("<I", f.read(4))[0]
        reloc_count = (packed >> 8) & 0xFFFFFF
        header_size = 0x18 + reloc_count * 8

        def walk_node(node_offset, depth=0):
            node = read_octree_sphere_node(f, node_offset + header_size)
            if node["strip"] > 0:
                indices = read_strip_indices(f, node["strip"], header_size)
                face_indices.extend(convert_indices_to_tris(indices))
            if depth < 2:
                for child_offset in node["spheres"]:
                    if child_offset > 0:
                        walk_node(child_offset, depth + 1)

        walk_node(octree_ptr)

    print(f"[Octree] Parsed {len(face_indices)} faces from section {section_index}")
    return face_indices

class TR7AE_OT_ImportLevel(bpy.types.Operator):
    bl_idname = "tr7ae.import_level"
    bl_label = "Import TR7AE Level"
    bl_description = "Import a Tomb Raider 7AE Level"
    bl_options = {'REGISTER', 'UNDO'}

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    filter_glob: bpy.props.StringProperty(
        default="*.level",
        options={'HIDDEN'},
        maxlen=255,
    )

    def execute(self, context):
        try:
            self.read_level_file(Path(self.filepath))
            self.report({'INFO'}, f"Imported TR7AE Level: {self.filepath}")
            return {'FINISHED'}
        except Exception as e:
            self.report({'ERROR'}, f"Failed to import level: {e}")
            return {'CANCELLED'}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def read_level_file(self, filepath: Path):
        with open(filepath, "rb") as f:
            assert f.read(4) == b'SECT', "Invalid file: missing SECT header"

            f.seek(0xC)
            packed_data = struct.unpack("<I", f.read(4))[0]
            num_relocs = (packed_data >> 8) & 0xFFFFFF

            section_id = struct.unpack("<I", f.read(4))[0]
            spec_mask = struct.unpack("<I", f.read(4))[0]

            relocations = []
            for _ in range(num_relocs):
                type_and_section = struct.unpack("<H", f.read(2))[0]
                reloc_type = type_and_section & 0b111
                section_index = type_and_section >> 3
                type_specific = struct.unpack("<H", f.read(2))[0]
                offset = struct.unpack("<I", f.read(4))[0]
                relocations.append({
                    "type": reloc_type,
                    "section": section_index,
                    "type_specific": type_specific,
                    "offset": offset
                })

            terrain_ptr = struct.unpack("<I", f.read(4))[0]
            print(f"[LEVEL] Terrain pointer: {terrain_ptr:#x}")

        base_reloc = next((r for r in relocations if r["offset"] == 0), None)
        if not base_reloc:
            raise RuntimeError("Missing relocation at offset 0.")

        section_index = base_reloc["section"]
        section_path = filepath.parent / f"{section_index}_0.gnc"

        if not section_path.exists():
            raise FileNotFoundError(f"Missing section file: {section_path.name}")

        with open(section_path, "rb") as sf:
            assert sf.read(4) == b'SECT', f"{section_path.name} is missing SECT header"

            # Read header and relocations from the .gnc file
            sf.seek(0xC)
            packed_data = struct.unpack("<I", sf.read(4))[0]
            gnc_num_relocs = (packed_data >> 8) & 0xFFFFFF
            gnc_header_size = 0x18 + (gnc_num_relocs * 8)

            # Parse relocations from the .gnc file (not the .level file!)
            sf.seek(0x18)
            gnc_relocations = []
            for _ in range(gnc_num_relocs):
                type_and_section = struct.unpack("<H", sf.read(2))[0]
                reloc_type = type_and_section & 0b111
                section_index = type_and_section >> 3
                type_specific = struct.unpack("<H", sf.read(2))[0]
                offset = struct.unpack("<I", sf.read(4))[0]
                gnc_relocations.append({
                    "type": reloc_type,
                    "section": section_index,
                    "type_specific": type_specific,
                    "offset": offset
                })

            sf.seek(terrain_ptr + gnc_header_size)
            data = sf.read(104)
            if len(data) < 104:
                raise ValueError("Terrain struct is incomplete or corrupt.")

            (
                unit_change_flags, spad,
                num_intros, intro_list,
                num_portals, portal_list,
                num_terrain_groups, terrain_groups_ptr,
                signal_terrain_group, signals,
                terrain_anim_textures,
                num_bg_instances, bg_instance_list,
                num_bg_objects, bg_object_list,
                num_vm_objects, vm_object_list,
                vm_table_list,
                xbox_pc_vb, xbox_pc_vmo, xbox_pc_portal_adjust,
                d3d_vertex_buffer, num_terrain_vertices,
                d3d_vmo_vertex_buffer, num_terrain_vmo_vertices,
                cdc_render_data_id, cdc_render_terrain_data
            ) = struct.unpack("<2h 22i 3I", data)

        print(f"[LEVEL] Terrain Struct:")
        print(f"  Num Intros: {num_intros}, Intro List Offset: {intro_list:#x}")
        print(f"  Terrain Vertices: {num_terrain_vertices}")
        print(f"  VMO Vertices: {num_terrain_vmo_vertices}")
        print(f"  CDC Render Data ID: {cdc_render_data_id:#x}")

        terrain_groups_ptr_offset = terrain_ptr + 24
        terrain_groups_reloc = next((r for r in gnc_relocations if r["offset"] == terrain_groups_ptr_offset), None)
        if not terrain_groups_reloc:
            raise RuntimeError(f"No relocation found for terrainGroups field at offset {terrain_groups_ptr_offset:#x}")

        tg_section_index = terrain_groups_reloc["section"]
        tg_section_path = filepath.parent / f"{tg_section_index}_0.gnc"
        if not tg_section_path.exists():
            raise FileNotFoundError(f"Missing section file for terrain groups: {tg_section_path.name}")

        terrain_group_entries = []

        with open(tg_section_path, "rb") as tf:
            assert tf.read(4) == b'SECT', f"{tg_section_path.name} missing SECT header"
            tf.seek(0xC)
            packed_data = struct.unpack("<I", tf.read(4))[0]
            tg_num_relocs = (packed_data >> 8) & 0xFFFFFF
            tf.seek(0x18)
            tg_relocations = []
            for _ in range(tg_num_relocs):
                type_and_section = struct.unpack("<H", tf.read(2))[0]
                reloc_type = type_and_section & 0b111
                section_index = type_and_section >> 3
                type_specific = struct.unpack("<H", tf.read(2))[0]
                offset = struct.unpack("<I", tf.read(4))[0]
                tg_relocations.append({
                    "type": reloc_type,
                    "section": section_index,
                    "type_specific": type_specific,
                    "offset": offset
                })
            tg_header_size = 0x18 + tg_num_relocs * 8

            tf.seek(terrain_groups_ptr + tg_header_size)

            for i in range(num_terrain_groups):
                raw = tf.read(176)
                if len(raw) < 176:
                    raise ValueError(f"Incomplete TerrainGroup at index {i}")

                octreeSphere_offset = 68
                octreeSphere_ptr = struct.unpack_from("<i", raw, octreeSphere_offset)[0]

                field_offset = terrain_groups_ptr + i * 176 + octreeSphere_offset
                reloc_offset = field_offset - tg_header_size
                octree_reloc = next((r for r in tg_relocations if r["offset"] == reloc_offset), None)
                print(f"[DEBUG] TerrainGroup[{i}] octreeSphere field file offset: 0x{field_offset:X} (value={octreeSphere_ptr:#x})")

                octree_reloc = next((r for r in tg_relocations if r["offset"] == field_offset), None)
                if not octree_reloc:
                    print(f"[TG] Skipping TerrainGroup[{i}]: no relocation for octreeSphere pointer field at offset 0x{field_offset:X}")
                    continue

                reloc_section = octree_reloc["section"]
                reloc_path = filepath.parent / f"{reloc_section}_0.gnc"

                terrain_group_entries.append({
                    "index": i,
                    "octreeSphere": octreeSphere_ptr,
                    "field_offset": field_offset,
                    "reloc_section": reloc_section,
                    "reloc_path": reloc_path
                })


        print(f"[TG] Loaded {len(terrain_group_entries)} terrain groups.")
        print(f"[TG] First group octreeSphere pointer: {terrain_group_entries[0]['octreeSphere']:#x}")

        face_indices = []

        for group in terrain_group_entries:
            octree_ptr = group["octreeSphere"]
            section_index = group["reloc_section"]

            try:
                faces = traverse_octree(filepath, octree_ptr, section_index)
                face_indices.extend(faces)
            except Exception as e:
                print(f"[Octree] Error loading group {group['index']} from section {section_index}: {e}")


        xbox_pc_vb_field_offset = terrain_ptr + 68

        vb_reloc = next((r for r in gnc_relocations if r["offset"] == xbox_pc_vb_field_offset), None)
        if not vb_reloc:
            raise RuntimeError(f"No relocation found for offset {xbox_pc_vb_field_offset:#x} (xbox_pc_vb position)")

        vb_section_index = vb_reloc["section"]
        vb_section_path = filepath.parent / f"{vb_section_index}_0.gnc"
        if not vb_section_path.exists():
            raise FileNotFoundError(f"Missing section file for xbox_pc_vb: {vb_section_path.name}")

        with open(vb_section_path, "rb") as vf:
            assert vf.read(4) == b'SECT', f"{vb_section_path.name} missing SECT header"

            vf.seek(0xC)
            packed_data = struct.unpack("<I", vf.read(4))[0]
            vb_num_relocs = (packed_data >> 8) & 0xFFFFFF
            vb_header_size = 0x18 + vb_num_relocs * 8

            vf.seek(xbox_pc_vb + 24)
            print(f"[VB] Pointer value xbox_pc_vb={xbox_pc_vb:#x} in section {vb_section_index}")

            vertex_stride = 20
            vertex_data = []

            for i in range(num_terrain_vertices):
                raw = vf.read(vertex_stride)
                if len(raw) < vertex_stride:
                    raise ValueError(f"Incomplete vertex at index {i}")

                (
                    x, y, z, w,
                    b, g, r, a,
                    u, v,
                    bend, bendindex
                ) = struct.unpack("<4h4B2h2H", raw)

                vertex_data.append({
                    "position": (x, y, z, w),
                    "color": (r, g, b, a),
                    "uv": (u, v),
                    "bend": bend,
                    "bendindex": bendindex
                })

            print(f"[VB] Loaded {len(vertex_data)} terrain vertices.")
            print(f"[VB] First vertex: {vertex_data[0]}")

            mesh = bpy.data.meshes.new("TerrainMesh")
            obj = bpy.data.objects.new("TerrainObject", mesh)
            bpy.context.collection.objects.link(obj)

            verts = [(v["position"][0] / 100.0, v["position"][2] / 100.0, -v["position"][1] / 100.0) for v in vertex_data]

            max_index = len(verts) - 1
            valid_faces = [face for face in face_indices if all(0 <= idx <= max_index for idx in face)]

            mesh.from_pydata(verts, [], valid_faces)
            mesh.update()

            color_layer = mesh.vertex_colors.new(name="Col")
            for poly in mesh.polygons:
                for loop_index in poly.loop_indices:
                    vertex_index = mesh.loops[loop_index].vertex_index
                    vcolor = vertex_data[vertex_index]["color"]
                    color_layer.data[loop_index].color = (
                        vcolor[0] / 255.0,
                        vcolor[1] / 255.0,
                        vcolor[2] / 255.0,
                        vcolor[3] / 255.0
                    )

            uv_layer = mesh.uv_layers.new(name="UVMap")
            for poly in mesh.polygons:
                for loop_index in poly.loop_indices:
                    vertex_index = mesh.loops[loop_index].vertex_index
                    u, v = vertex_data[vertex_index]["uv"]
                    uv_layer.data[loop_index].uv = (
                        u / 4096.0,
                        1.0 - (v / 4096.0)
                    )

            print(f"[Blender] Imported {len(verts)} vertices and {len(valid_faces)} faces into mesh.")


def register():
    classes = [
        ClothPointData, DistRuleData, ClothJointMapData,
        TR7AE_SphereInfo, TR7AE_HCapsuleInfo, TR7AE_HBoxInfo, TR7AE_HMarkerInfo, TR7AE_OT_ImportNextGenModel, TR7AE_OT_ExportNextGenModel, ImportTR7AEPS2, ImportPBRWC,
        TR7AE_OT_ImportModel, TR7AE_PT_Tools, TR7AE_OT_ToggleHSpheres, TR7AE_OT_ImportLevel, TR7AE_AnimationSettings,
        TR7AE_OT_ToggleHBoxes, TR7AE_OT_ToggleHMarkers, TR7AE_PT_DrawGroupPanel, TR7AE_OT_ImportAnimation,  ImportUnderworldGOLModel, ExportUnderworldModel, ImportUnderworldGOLAnimation, ExportUnderworldAnimation,
        TR7AE_PT_ModelDebugProperties, TR7AE_PT_MaterialPanel, NextGenMaterialProperties, NextGenMaterialPanel, TR7AE_PT_SphereInfo, TR7AE_OT_ExportAnimation,
        TR7AE_PT_HCapsuleInfo, TR7AE_PT_HBoxInfo, TR7AE_PT_MarkersInfo, TR7AE_PT_ClothPointInspector, TR7AE_PT_DistRuleInspector, TR7AE_PT_ClothJointMapInspector,
        TR7AE_OT_NormalizeAndLimitWeights, TR7AE_OT_ToggleMarker, TR7AE_OT_ToggleHBox, TR7AE_OT_AddHBox, TR7AE_OT_AddHCapsule,
        TR7AE_OT_ToggleSphere, TR7AE_OT_ToggleCapsule, TR7AE_OT_ToggleBox, TR7AE_OT_AddHSphere, TR7AE_OT_AddHMarker,
        TR7AE_PT_HMarkerMeshInfo, TR7AE_PT_HCapsuleMeshInfo, TR7AE_OT_ToggleHCapsules, TR7AE_PT_AddHInfoButtons,
        TR7AE_OT_ExportOldGenModel, TR7AE_OT_SnapBoneToHMarker, TR7AE_OT_ConvertImageToPCD, TR7AE_PT_TextureTools,
        TR7AE_PT_ClothPanel, TR7AE_ClothSettings, TR7AE_ModelTargetInfo, TR7AE_OT_ConvertPCDToImage, TR7AE_OT_ConvertRAWToImage, TR7AE_OT_ConvertImageToRAW,
        TR7AE_PT_HSphereMeshInfo, TR7AE_PT_UtilitiesPanel, TR7AE_Preferences, TR7AE_OT_ClearTextureCache,
        TR7AE_PT_VisibilityPanel, TR7AE_SectionPaths, TR7AE_OT_ToggleHInfoVisibility,
        TR7AE_PT_FileSectionsPanel
    ]

    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.Object.max_rad = bpy.props.FloatProperty(
        name="Max Radius",
        description="Defines the approximate maximum extent of the model from its origin, used for view frustum culling",
        default=0.0
    )

    bpy.types.Object.cdcRenderDataID = bpy.props.IntProperty(
        name="cdcRenderDataID",
        description="Model ID, used for the game to recognize and replace with Next Gen model.\n\nOnly actually utilized for Legend, always 0 for Anniversary",
        default=0
    )

    bpy.types.PoseBone.tr7ae_hmarkers = bpy.props.CollectionProperty(type=TR7AE_HMarkerInfo)
    bpy.types.PoseBone.tr7ae_hcapsules = bpy.props.CollectionProperty(type=TR7AE_HCapsuleInfo)
    bpy.types.PoseBone.tr7ae_hspheres = bpy.props.CollectionProperty(type=TR7AE_SphereInfo)
    bpy.types.PoseBone.tr7ae_hboxes = bpy.props.CollectionProperty(type=TR7AE_HBoxInfo)
    bpy.types.Object.cloth_points = bpy.props.CollectionProperty(type=ClothPointData)
    bpy.types.Object.tr7ae_distrules = bpy.props.CollectionProperty(type=DistRuleData)
    bpy.types.Object.tr7ae_jointmaps = bpy.props.CollectionProperty(type=ClothJointMapData)
    bpy.types.Action.tr7ae_anim_settings = bpy.props.PointerProperty(type=TR7AE_AnimationSettings)
    bpy.types.Material.nextgen_material_properties = bpy.props.PointerProperty(type=NextGenMaterialProperties)
    bpy.types.Object.tr7ae_sections = bpy.props.PointerProperty(type=TR7AE_SectionPaths)
    bpy.types.Object.tr7ae_modeltarget = bpy.props.PointerProperty(type=TR7AE_ModelTargetInfo)
    bpy.types.Object.tr7ae_cloth = bpy.props.PointerProperty(type=TR7AE_ClothSettings)

    register_envmap_property()
    register_eyerefenvmap_property()
    register_draw_group_property()
    register_material_properties()
    register_panel_properties()

    handlers = [
        tr7ae_sync_handler, sync_model_target_properties,
        sync_hmarker_transforms, sync_hbox_transforms,
        sync_hcapsule_transforms
    ]

    if simulate_cloth_on_frame not in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.append(simulate_cloth_on_frame)

    for handler in handlers:
        if handler not in bpy.app.handlers.depsgraph_update_post:
            bpy.app.handlers.depsgraph_update_post.append(handler)



def unregister():
    handlers = [
        tr7ae_sync_handler, sync_model_target_properties,
        sync_hmarker_transforms, sync_hbox_transforms,
        sync_hcapsule_transforms
    ]

    for handler in handlers:
        if handler in bpy.app.handlers.depsgraph_update_post:
            bpy.app.handlers.depsgraph_update_post.remove(handler)

    del bpy.types.PoseBone.tr7ae_hspheres
    del bpy.types.PoseBone.tr7ae_hcapsules
    del bpy.types.PoseBone.tr7ae_hboxes
    del bpy.types.PoseBone.tr7ae_hmarkers
    del bpy.types.Object.cloth_points
    del bpy.types.Object.tr7ae_distrules
    del bpy.types.Object.tr7ae_jointmaps
    del bpy.types.Object.max_rad
    del bpy.types.Object.cdcRenderDataID
    del bpy.types.Action.tr7ae_anim_settings

    del bpy.types.Object.tr7ae_sections
    del bpy.types.Object.tr7ae_modeltarget
    del bpy.types.Object.tr7ae_cloth

    unregister_envmap_property()
    unregister_draw_group_property()
    unregister_material_properties()
    unregister_panel_properties()

    classes = [
        TR7AE_PT_FileSectionsPanel, TR7AE_SectionPaths,
        TR7AE_PT_VisibilityPanel, TR7AE_PT_UtilitiesPanel,
        TR7AE_PT_HSphereMeshInfo, TR7AE_ModelTargetInfo, TR7AE_OT_ImportNextGenModel, TR7AE_OT_ExportNextGenModel, ImportTR7AEPS2, ImportPBRWC,
        TR7AE_ClothSettings, TR7AE_PT_ClothPanel, TR7AE_OT_ImportLevel, TR7AE_OT_ImportAnimation,
        TR7AE_OT_SnapBoneToHMarker, TR7AE_PT_HCapsuleMeshInfo, ImportUnderworldGOLModel, ExportUnderworldModel, ImportUnderworldGOLAnimation, ExportUnderworldAnimation,
        TR7AE_PT_HMarkerMeshInfo, TR7AE_OT_ToggleBox, TR7AE_AnimationSettings,
        TR7AE_OT_ToggleCapsule, TR7AE_OT_ToggleSphere, TR7AE_PT_ClothPointInspector, TR7AE_PT_DistRuleInspector, TR7AE_PT_ClothJointMapInspector,
        TR7AE_OT_ToggleMarker, TR7AE_PT_MarkersInfo, TR7AE_OT_AddHSphere, TR7AE_OT_ExportAnimation,
        TR7AE_OT_ExportOldGenModel, TR7AE_OT_NormalizeAndLimitWeights, TR7AE_OT_AddHCapsule,
        TR7AE_PT_HBoxInfo, TR7AE_PT_HCapsuleInfo, TR7AE_OT_ToggleHBox, TR7AE_OT_AddHMarker, TR7AE_OT_AddHBox,
        TR7AE_PT_SphereInfo, TR7AE_PT_MaterialPanel, NextGenMaterialProperties, NextGenMaterialPanel, TR7AE_OT_ConvertImageToPCD, TR7AE_PT_TextureTools,
        TR7AE_PT_ModelDebugProperties, TR7AE_PT_DrawGroupPanel, TR7AE_OT_ConvertPCDToImage, TR7AE_OT_ConvertRAWToImage, TR7AE_OT_ConvertImageToRAW, TR7AE_PT_AddHInfoButtons,
        TR7AE_OT_ToggleHMarkers, TR7AE_OT_ToggleHBoxes, TR7AE_Preferences, TR7AE_OT_ClearTextureCache,
        TR7AE_OT_ToggleHSpheres, TR7AE_PT_Tools, TR7AE_OT_ToggleHCapsules,
        TR7AE_OT_ImportModel, TR7AE_HMarkerInfo, TR7AE_OT_ToggleHInfoVisibility,
        TR7AE_HBoxInfo, TR7AE_HCapsuleInfo, TR7AE_SphereInfo,
        ClothPointData, DistRuleData, ClothJointMapData
    ]

    if simulate_cloth_on_frame in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.remove(simulate_cloth_on_frame)

    for cls in classes:
        bpy.utils.unregister_class(cls)

if __name__ == "__main__":
    register()