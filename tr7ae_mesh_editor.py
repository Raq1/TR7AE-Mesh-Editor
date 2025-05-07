bl_info = {
    "name": "Tomb Raider 7/AE Mesh Editor",
    "author": "Raq",
    "collaborators": "Che, TheIndra, arcusmaximus (arc), DKDave, Joschka, Henry",
    "version": (1, 1, 5),
    "blender": (4, 2, 3),
    "location": "View3D > Sidebar > TR7AE Tools",
    "description": "Import and export Tomb Raider Legend/Anniversary mesh files.",
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

def convert_dds_to_pcd(dds_path, output_pcd_path):
    dds_path = Path(dds_path)
    with open(dds_path, "rb") as f:
        header = f.read(128)
        if header[:4] != b'DDS ':
            raise ValueError("Invalid DDS file")

        height = struct.unpack_from("<I", header, 12)[0]
        width = struct.unpack_from("<I", header, 16)[0]
        mipmaps = struct.unpack_from("<I", header, 28)[0]
        fourcc = header[84:88]
        dxt_data = f.read()

    format_bytes = struct.unpack("<I", fourcc)[0]

    # Compute texture ID from filename (assumes format INDEX_ID.dds)
    try:
        texture_name = dds_path.stem  # '243_1c'
        texture_id_hex = texture_name.split('_')[1]  # '1c'
        texture_id = int(texture_id_hex, 16)
    except Exception:
        raise ValueError("Failed to parse texture ID from filename. Use format INDEX_ID.dds")

    total_data_size = 24 + len(dxt_data)  # Correct total data size

    magic = 0x39444350  # 'PCD9'
    pcd_header = struct.pack(
        "<I i I I H H B B H",
        magic,
        format_bytes,         # DDS type (from FourCC)
        len(dxt_data),        # bitmap size
        0,                    # unknown
        width,
        height,
        0,                    # unknown
        mipmaps - 1,
        3                     # unknown
    )

    with open(output_pcd_path, "wb") as out:
        # Write SECT header (24 bytes)
        out.write(struct.pack("<4sI", b'SECT', total_data_size))  # Magic + total size
        out.write(struct.pack("<B", 5))  # Type
        out.write(b"\x00" * 7)  # Reserved
        out.write(struct.pack("<I", texture_id))  # Texture ID
        out.write(struct.pack("<I", 0xFFFFFFFF))  # SpecMask

        # Write the .PCD block
        out.write(pcd_header)
        out.write(dxt_data)      # Only the raw texture data, excluding FourCC again


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

bpy.types.Object.tr7ae_model_scale = bpy.props.FloatVectorProperty(
    name="Model Scale",
    size=3,
    subtype='XYZ',
    description="Model Scale imported from the file.\n\nBecause the format stores vertex positions in shorts,\nthey are scaled using these fields to make sure the data\nfits within what can be stored in a signed short.\n\nIf your model is relatively big compared to the original,\nyou need to use bigger scale values\nfor your model to write correctly, without looking ""squished"".\n\nIf your model is relatively small compared to the original,\nyou can use smaller scale values\nfor more accurate vertex positions.\n\nIf unsure, simply leave this field default, and make sure\nyour model fits the proportions of the original"
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

    # 1) Gather all “Mesh_*” children
    meshes = [obj for obj in armature.children
            if obj.type == 'MESH'
            and not obj.get("tr7ae_is_mface")]
    if not meshes:
        return [], {}

    # 2) Map bone names → indices
    bone_indices = {b.name: i for i, b in enumerate(armature.data.bones)}

    # 3) Prepare per‑bone lists for single‑weight verts
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
                raise ValueError(f"Mesh '{mesh.name}' has more than 2 weights per vertex.\n Use Limit Weights and Normalize All.")

            if len(bone_ids) == 1:
                single_by_bone[bone_ids[0]].append((mesh, v.index, bone_ids))
            elif len(bone_ids) == 2:
                two_weight.append((mesh, v.index, bone_ids, bone_weight_pairs))

        eval_obj.to_mesh_clear()

    # 4) Flatten: all single-weight by bone order, then all two-weight (sorted by bone pairs)
    sorted_verts = []
    bone_ranges = {}
    cursor = 0

    # Group 1-weighted verts
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
        w0 = bone_weights[0][1]  # weight of b0
        q_weight = round(w0 * 15) / 15
        return (b0, b1, q_weight, mesh.name, v_idx)

    two_weight_sorted = sorted(two_weight, key=sort_key)

    # And strip the weights back out (if needed downstream):
    two_weight_sorted = [(mesh, v_idx, bone_ids) for mesh, v_idx, bone_ids, _ in two_weight_sorted]


    # Append them
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

        # ── HSpheres ──
        for s in getattr(pbone, "tr7ae_hspheres", []):
            if s.radius <= 0:
                continue
            entry.spheres.append((
                int(s.flags),
                int(s.id),
                int(s.rank),
                int(round(s.radius)),                      # scale → int
                int(round(s.x)), int(round(s.y)), int(round(s.z)),  # pos → ints
                int(round(s.radius_sq)),
                int(round(s.mass)),
                int(round(s.buoyancy_factor)),
                int(round(s.explosion_factor)),
                int(round(s.material_type)),
                int(round(s.pad)),
                int(round(s.damage)),
            ))

        # ── HBoxes ──
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

        # ── HCapsules ──
        for c in getattr(pbone, "tr7ae_hcapsules", []):
            if c.flags == 0:
                continue
            entry.hcapsules.append((
                c.posx,    # pos → int
                c.posy,
                c.posz,
                c.posw,
                c.quatx, c.quaty, c.quatz, c.quatw,
                int(c.flags),
                int(c.id),
                int(c.rank),
                int(round(c.radius)),  # scale → int
                int(round(c.length)),  # scale → int
                int(round(c.mass)),
                int(round(c.buoyancy_factor)),
                int(round(c.explosion_factor)),
                int(c.material_type),
                int(c.pad),
                int(c.damage),
            ))

        # Markers: This assumes a list stored on the pose bone
        marker_list = getattr(pbone, "tr7ae_hmarkers", [])
        for m in marker_list:
            marker_obj = bpy.data.objects.get(f"HMarker_{m.bone}_{m.index}")
            if marker_obj:
                # Use the current world-space rotation from the object
                euler = marker_obj.rotation_euler
                rx, ry, rz = euler.x, euler.y, euler.z
            else:
                # Fallback to the stored panel values
                rx, ry, rz = m.rx, m.ry, m.rz

            entry.hmarkers.append((
                m.bone, m.index,
                m.px, m.py, m.pz,
                rx, ry, rz
            ))

        # only include bones that have any data
        if entry.spheres or entry.hboxes or entry.hcapsules or entry.hmarkers:
            entries.append(entry)

    return entries

def align_stream(mb, alignment):
    padding = (alignment - (mb.tell() % alignment)) % alignment
    if padding:
        mb.write(b"\x00" * padding)

class TR7AE_OT_ExportCustomModel(Operator):
    bl_idname = "tr7ae.export_custom_model"
    bl_label = "Export TR7AE Model"
    bl_description = "Export the model in TR7AE format"
    bl_options = {'REGISTER'}

    filepath: StringProperty(subtype="FILE_PATH")

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
                    offset = mb.tell()  # relative to start of mb
                    reloc_list.append(offset)
                    mb.write(struct.pack("<I", value))

                # --- Header Part 2: Start of Model Header ---
                sorted_verts, bone_vertex_ranges = collect_and_sort_mesh_vertices(armature)
                if len(sorted_verts) > 21845:
                    self.report({'WARNING'}, f"Model has {len(sorted_verts)} vertices — game limit is 21845!")
                collection = bpy.context.view_layer.active_layer_collection.collection
                mesh_objs = [
                    obj for obj in collection.objects
                    if obj.parent == armature and obj.type == 'MESH'
                ]
                vertex_index_map = {
                    (mesh.name, orig_idx): i
                    for i, (mesh, orig_idx, _) in enumerate(sorted_verts)
                }
                # Step 1: Detect if Flat Shading is used
                flat_shading_used = False
                collection = bpy.context.view_layer.active_layer_collection.collection
                mesh_objs = [
                    obj for obj in collection.objects
                    if obj.parent == armature and obj.type == 'MESH'
                ]

                for mesh_obj in mesh_objs:
                    if mesh_obj.material_slots:
                        for slot in mesh_obj.material_slots:
                            material = slot.material
                            if material and material.get("tr7ae_flat_shading", 0) == 1:
                                flat_shading_used = True
                                break
                    if flat_shading_used:
                        break

                # Step 2: Generate virt_entries, quantizing until count <= 153
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
                    # else: keep the unquantized result (already valid)

                else:
                    # Use full precision
                    virt_entries = collect_virtsegment_entries(sorted_verts, armature, quantize=False)


                # Maps vertex index in sorted_verts → virtual segment index
                virt_segment_lookup = {}

                base_index = len(armature.data.bones)  # starting segment index for virtual segments

                for i, (start, end, b0, b1, w) in enumerate(virt_entries):
                    for v in range(start, end + 1):
                        virt_segment_lookup[v] = base_index + i

                num_virtsegments = len(virt_entries)

                version = 79823955
                bones = armature.data.bones
                num_bones = len(bones)
                bone_data_offset_pos = f.tell() + 12  # position of the 4th int (after version, num_bones, num_virtsegments)
                mb.write(struct.pack("<4I", version, num_bones, num_virtsegments, 0))  # write 0 for now

                model_scale = getattr(armature, "tr7ae_model_scale", (1.0, 1.0, 1.0))
                scale = (*model_scale, 1.0)
                mb.write(struct.pack("<4f", *scale))

                max_rad = armature.get("max_rad", 0.0)
                max_rad_sq = armature.get("max_rad_sq", max_rad * max_rad)
                cdcRenderDataID = armature.get("cdcRenderDataID")

                # Placeholder values for rest of the header
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

                    # Optional: null terminator (depends on engine needs)
                    mb.write(struct.pack("<H", 0))

                # --- Bone Data Section ---
                sorted_verts, bone_vertex_ranges = collect_and_sort_mesh_vertices(armature)
                align_stream(mb, 16)
                bone_offset = mb.tell()
                mb.seek(bone_data_offset_pos)
                write_offset(mb, bone_offset, relocations)
                mb.seek(bone_offset)  # go back to actually write bones
                bone_index_map = {b.name: i for i, b in enumerate(bones)}

                # Collect HInfo entries BEFORE writing bones
                hinfo_entries = collect_structured_hinfo_data(armature)

                # Build map for patching later
                hentry_by_index = {e.bone_index: e for e in hinfo_entries}

                # Step 1: Write bones with placeholder HInfo offset
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


                # 2) build the virtsegment entries
                align_stream(mb, 16)
                num_virtsegments = len(virt_entries)

                # 3) write your header (update num_virtsegments)
                #    at the point where you do:
                #       mb.write(struct.pack("<4I",
                #           version,
                #           num_bones,
                #           num_virtsegments,       # <-- now correct
                #           bone_data_offset))
                #    …then write bones as before.

                # 4) immediately after your bone‑data loop, write:
                for first, last, b0, b1, w in virt_entries:
                    # Write two dummy blocks
                    mb.write(struct.pack("<4f", 0.0, 0.0, 0.0, 0.0))
                    mb.write(struct.pack("<4f", 0.0, 0.0, 0.0, 0.0))

                    # Use the cached bone position and negate it
                    def compute_virtual_segment_pivot(b0, b1, bones, write_bone_index):
                        bone_a = bones[b0]
                        bone_b = bones[b1]

                        # Decide which one is the "write target"
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

                # Step 3: Write HInfo blocks and store offsets
                for hentry in hinfo_entries:
                    hentry.offset = mb.tell()

                    # Placeholder positions for list offsets
                    hsphere_offset_pos = mb.tell() + 4
                    hbox_offset_pos = hsphere_offset_pos + 8
                    hmarker_offset_pos = hbox_offset_pos + 8
                    hcapsule_offset_pos = hmarker_offset_pos + 8

                    # Write counts and dummy offsets (will be patched)
                    mb.write(struct.pack("<iI", len(hentry.spheres), 0))  # HSpheres
                    mb.write(struct.pack("<iI", len(hentry.hboxes), 0))   # HBoxes
                    mb.write(struct.pack("<iI", len(hentry.hmarkers), 0)) # HMarkers
                    mb.write(struct.pack("<iI", len(hentry.hcapsules), 0))# HCapsules

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
                    write_list(hentry.hcapsules, "<4f4f h bb hhhbbbb h ", hcapsule_offset_pos)

                # Step 4: Patch bones with real HInfo offsets
                for hentry in hinfo_entries:
                    bone_struct_start = bone_offset + hentry.bone_index * 64
                    hinfo_field_offset = bone_struct_start + 60  # HInfo is at byte 60
                    mb.seek(hinfo_field_offset)
                    write_offset(mb, hentry.offset, relocations)

                mb.seek(0, 2)  # Seek to end of file to append the vertex list cleanly
                vertex_list_offset = mb.tell()  # where the actual vertex list starts
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

                # Ensure all meshes are triangulated if necessary
                for obj in bpy.context.view_layer.objects:
                    if obj.type != 'MESH' or obj.parent != armature:
                        continue

                    depsgraph = bpy.context.evaluated_depsgraph_get()
                    eval_obj = obj.evaluated_get(depsgraph)
                    eval_mesh = eval_obj.to_mesh()

                    needs_triangulate = any(poly.loop_total != 3 for poly in eval_mesh.polygons)

                    eval_obj.to_mesh_clear()

                    if needs_triangulate:
                        print(f"[INFO] Adding Triangulate modifier to '{obj.name}' (non-triangular faces detected).")
                        triangulate_mod = obj.modifiers.new(name="TRIANGULATE_TEMP", type='TRIANGULATE')
                        triangulate_mod.show_expanded = False  # Optional: Hide it in the UI

                # Ensure all meshes have vertex colors (add white if missing)
                for obj in bpy.context.view_layer.objects:
                    if obj.type != 'MESH' or obj.parent != armature:
                        continue

                    mesh = obj.data
                    if not mesh.color_attributes:
                        print(f"[INFO] Adding Vertex Colors to '{obj.name}'.")
                        # Create a new vertex color layer named "Color" (Face Corner domain for compatibility)
                        color_layer = mesh.color_attributes.new(name="Color", type='BYTE_COLOR', domain='CORNER')

                        # Fill it with white (RGBA 255)
                        for color in color_layer.data:
                            color.color = (1.0, 1.0, 1.0, 1.0)

                for v_idx, (mesh, orig_idx, bone_ids) in enumerate(sorted_verts):
                    eval_obj = mesh.evaluated_get(depsgraph)
                    eval_mesh = eval_obj.to_mesh()
                    v = eval_mesh.vertices[orig_idx]

                    # Convert to world space
                    world_co = mesh.matrix_world @ v.co

                    # Determine primary bone index
                    bone_index = bone_ids[0] if bone_ids else 0
                    bone_name = f"Bone_{bone_index}"

                    # Convert to bone-local space
                    if bone_name in armature.data.bones:
                        bone = armature.data.bones[bone_name]
                        bone_world_matrix = armature.matrix_world @ bone.matrix_local
                        bone_local_co = bone_world_matrix.inverted() @ world_co
                    else:
                        bone_local_co = world_co

                    # Scale and clamp
                    model_scale = getattr(armature, "tr7ae_model_scale", (1.0, 1.0, 1.0))
                    scaled = Vector((
                        bone_local_co.x / model_scale[0],
                        bone_local_co.y / model_scale[1],
                        bone_local_co.z / model_scale[2]
                    ))
                    x = clamp_short(scaled.x)
                    y = clamp_short(scaled.y)
                    z = clamp_short(scaled.z)

                    # Write vertex colors
                    layer_color = mesh.data.vertex_colors.active
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


                    # Normals
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

                    # Segment
                    if v_idx in virt_segment_lookup:
                        segment = virt_segment_lookup[v_idx]
                    else:
                        segment = bone_ids[0] if bone_ids else 0
                    # UVs
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

                    # Write vertex
                    mb.write(struct.pack("<3h", x, y, z))       # position
                    mb.write(struct.pack("<3b", nx, ny, nz))    # normal
                    mb.write(struct.pack("<B", 0))              # pad
                    mb.write(struct.pack("<h", segment))        # bone index
                    mb.write(struct.pack("<2H", uvx, uvy))      # UV

                    eval_obj.to_mesh_clear()

                align_stream(mb, 16)
                mface_list_offset = mb.tell()  # SAVE START OF MFACE ONCE

                for mesh_obj in mesh_objs:
                    if not mesh_obj.get("tr7ae_is_mface"):
                        continue  # Only process MFace meshes

                    depsgraph = bpy.context.evaluated_depsgraph_get()
                    eval_obj = mesh_obj.evaluated_get(depsgraph)
                    eval_mesh = eval_obj.to_mesh()

                    vcol_layer = eval_mesh.vertex_colors.get("DryingGroups")
                    if not vcol_layer:
                        print(f"[WARNING] MFace {mesh_obj.name} missing DryingGroups.")
                        continue

                    # Build a position → index map from the real sorted verts
                    real_positions = {}
                    for idx, (mesh, orig_idx, _) in enumerate(sorted_verts):
                        eval_real_obj = mesh.evaluated_get(depsgraph)
                        eval_real_mesh = eval_real_obj.to_mesh()
                        real_vert = eval_real_mesh.vertices[orig_idx]
                        pos = (mesh.matrix_world @ real_vert.co).to_tuple(5)
                        real_positions[pos] = idx
                        eval_real_obj.to_mesh_clear()

                    # Build remap
                    vertex_remap = {}
                    for v in eval_mesh.vertices:
                        mface_pos = (mesh_obj.matrix_world @ v.co).to_tuple(5)

                        # Find exact match
                        target_idx = real_positions.get(mface_pos)
                        if target_idx is None:
                            raise ValueError(f"Vertex at {mface_pos} not found in sorted verts!")

                        vertex_remap[v.index] = target_idx

                    for poly in eval_mesh.polygons:
                        if poly.loop_total != 3:
                            continue  # Only triangles

                        v0 = vertex_remap[eval_mesh.loops[poly.loop_start + 0].vertex_index]
                        v1 = vertex_remap[eval_mesh.loops[poly.loop_start + 1].vertex_index]
                        v2 = vertex_remap[eval_mesh.loops[poly.loop_start + 2].vertex_index]

                        # Collect original group IDs
                        gid_lookup = {}
                        for i in range(3):
                            original_v = eval_mesh.loops[poly.loop_start + i].vertex_index
                            color = vcol_layer.data[poly.loop_start + i].color
                            gid = find_closest_group_id(color)
                            gid_lookup[vertex_remap[original_v]] = gid

                        # Now use remapped indices to build 'same'
                        same = (gid_lookup[v2] << 10) | (gid_lookup[v1] << 5) | gid_lookup[v0]

                        mb.write(struct.pack("<4H", v0, v1, v2, same))

                    eval_obj.to_mesh_clear()

                # === Patch MFace pointer ===
                for mesh_obj in mesh_objs:
                    if mesh_obj.get("tr7ae_is_mface"):
                        mb.seek(num_faces_offset + 4)
                        write_offset(mb, mface_list_offset, relocations)
                        mb.seek(0, 2)

                align_stream(mb, 16)
                face_list_offset = mb.tell()

                # Patch the header
                mb.seek(face_list_offset_pos)
                write_offset(mb, face_list_offset, relocations)
                mb.seek(face_list_offset)

                chunk_starts = []

                total_faces = 0

                envmapped_vertex_indices = set()
                eye_ref_vertex_indices = set()

                for mesh_obj in mesh_objs:
                    if mesh_obj.get("tr7ae_is_mface"):
                        continue  # Skip MFace-only meshes!
                    depsgraph = bpy.context.evaluated_depsgraph_get()
                    eval_obj = mesh_obj.evaluated_get(depsgraph)
                    eval_mesh = eval_obj.to_mesh()

                    indices = []

                    for poly in eval_mesh.polygons:
                        if poly.loop_total != 3:
                            continue
                        tri = []
                        for i in range(3):
                            loop_idx = poly.loop_start + i
                            v_idx = eval_mesh.loops[loop_idx].vertex_index
                            mapped_idx = vertex_index_map.get((mesh_obj.name, v_idx))
                            if mapped_idx is not None:
                                tri.append(mapped_idx)
                        if len(tri) == 3:
                            indices.extend(tri)

                            # These are written to GNC files
                            if mesh_obj.data.get("tr7ae_is_envmapped", False):
                                envmapped_vertex_indices.update(tri)
                                wrote_env_mapped = True
                            if mesh_obj.data.get("tr7ae_is_eyerefenvmapped", False):
                                eye_ref_vertex_indices.update(tri)
                                wrote_eye_ref = True

                    eval_obj.to_mesh_clear()

                    vertex_count = len(indices)
                    if vertex_count == 0:
                        continue

                    align_stream(mb, 16)
                    chunk_starts.append(mb.tell())

                    # Read draw_group from mesh custom property
                    draw_group = int(mesh_obj.data.get("tr7ae_draw_group", 0))

                    # Use the first material slot ONLY (since every mesh has one material)
                    material = None
                    if mesh_obj.material_slots:
                        # Get material index from the first polygon
                        poly_mat_index = mesh_obj.data.polygons[0].material_index if mesh_obj.data.polygons else 0
                        if poly_mat_index < len(mesh_obj.material_slots):
                            material = mesh_obj.material_slots[poly_mat_index].material

                    texture_id   = int(material.get("tr7ae_texture_id", 0)) if material else 0
                    blend_value  = int(material.get("tr7ae_blend_value", 0)) if material else 0
                    unknown_1    = int(material.get("tr7ae_unknown_1", 0)) if material else 0
                    unknown_2    = int(material.get("tr7ae_unknown_2", 0)) if material else 0
                    single_sided = int(material.use_backface_culling) if material else 1
                    texture_wrap    = int(material.get("tr7ae_texture_wrap", 0)) if material else 0
                    unknown_3    = int(material.get("tr7ae_unknown_3", 0)) if material else 0
                    unknown_4    = int(material.get("tr7ae_unknown_4", 0)) if material else 0
                    flat_shading    = int(material.get("tr7ae_flat_shading", 0)) if material else 0
                    sort_z    = int(material.get("tr7ae_sort_z", 0)) if material else 0
                    stencil_pass    = int(material.get("tr7ae_stencil_pass", 0)) if material else 0
                    stencil_func    = int(material.get("tr7ae_stencil_func", 0)) if material else 0
                    alpha_ref    = int(material.get("tr7ae_alpha_ref", 0)) if material else 0

                    # print(f"[DEBUG] Mesh: {mesh_obj.name}")
                    # print(f"[DEBUG] Material: {material.name if material else 'None'}")
                    # print(f"[DEBUG] texture_id={texture_id}, blend={blend_value}, unk1={unknown_1}, single_sided={single_sided}, unk2={unknown_2}")

                    tpageid = (
                        (texture_id   & 0x1FFF)      |  # bits 0–12
                        ((blend_value & 0xF)   << 13) |  # bits 13–16
                        ((unknown_1   & 0x7)   << 17) |  # bits 17–19
                        ((unknown_2   & 0x1)   << 20) |  # bit 20
                        ((single_sided & 0x1) << 21) |  # bit 21
                        ((texture_wrap & 0x3) << 22) |  # bits 22–23
                        ((unknown_3   & 0x1)   << 24) |  # bit 24
                        ((unknown_4   & 0x1)   << 25) |  # bit 25
                        ((flat_shading & 0x1) << 26) |  # bit 26
                        ((sort_z      & 0x1)   << 27) |  # bit 27
                        ((stencil_pass & 0x3) << 28) |  # bits 28–29
                        ((stencil_func & 0x1) << 30) |  # bit 30
                        ((alpha_ref   & 0x1)   << 31)   # bit 31
                    )

                    mb.write(struct.pack("<2h", vertex_count, draw_group))
                    mb.write(struct.pack("<I", tpageid))
                    mb.write(b"\x00" * 8)
                    mb.write(struct.pack("<I", 0))  # placeholder for next_chunk_ptr
                    face_count = vertex_count // 3
                    total_faces += face_count
                    mb.write(struct.pack(f"<{vertex_count}H", *indices))

                last_real_chunk_offset = chunk_starts[-1]
                terminator_offset = mb.tell()
                mb.write(b"\x00" * 0x14)  # 0x14 = size of chunk header without indices
                mb.seek(last_real_chunk_offset + 0x10)  # offset to next_chunk_ptr
                write_offset(mb, terminator_offset, relocations)

                for i in range(len(chunk_starts) - 1):
                    this_chunk_offset = chunk_starts[i]
                    next_chunk_offset = chunk_starts[i + 1]

                    # Seek to next_chunk_ptr location
                    mb.seek(this_chunk_offset + 0x10)  # ← 0x10 is correct offset
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

                # Write SECT header and section metadata
                model_data = bytearray(mb.getvalue())

                if wrote_vertex_colors:
                    vertex_color_offset_in_gnc = 8  # data in .gnc after dummy
                    struct.pack_into("<I", model_data, 100, vertex_color_offset_in_gnc)
                    relocations.append(100)
                model_data_size = len(model_data)

                with open(self.filepath, "wb") as f:
                    f.write(b'SECT')                                 # 0x00
                    f.write(struct.pack("<I", model_data_size))      # 0x04
                    f.write(struct.pack("<I", 0))                    # 0x08 (reserved)
                    packed_reloc = (len(relocations) << 8)
                    f.write(struct.pack("<I", packed_reloc))         # 0x0C
                    f.write(struct.pack("<I", 0))                    # 0x10 (reserved)
                    f.write(struct.pack("<I", 0xFFFFFFFF))           # 0x14 specializationMask
                    # Ensure we're padded to offset 100 for the vertex color pointer
                    while mb.tell() < 100:
                        mb.write(b"\x00")
                    mb.write(struct.pack("<I", 0))  # offset 100 — placeholder for vertex color offset


                    # Write relocations
                    # Access section indices from armature
                    sections = armature.tr7ae_sections

                    # Write relocations with real section index values
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
                            ef.write(b'SECT')                          # 0x00
                            total_gnc_size = 8 + len(gnc_data)
                            if wrote_env_mapped:
                                total_gnc_size += 4 + len(envmapped_vertex_indices) * 2
                            if wrote_eye_ref:
                                total_gnc_size += 4 + len(eye_ref_vertex_indices) * 2

                            ef.write(struct.pack("<I", total_gnc_size))  # 0x04
                            ef.write(struct.pack("<I", 0))             # 0x08
                            ef.write(struct.pack("<I", 0))             # 0x0C
                            ef.write(struct.pack("<I", 0))             # 0x10
                            ef.write(struct.pack("<I", 0xFFFFFFFF))    # 0x14
                            ef.write(struct.pack("<Q", 0))             # 0x18 dummy

                            vertex_color_offset_in_gnc = ef.tell()     # should now be 32

                            ef.write(gnc_data)

                            if wrote_env_mapped:
                                env_offset_in_gnc = ef.tell()  # ← Add this
                                env_list = sorted(envmapped_vertex_indices)
                                ef.write(struct.pack("<I", len(env_list)))
                                ef.write(struct.pack(f"<{len(env_list)}H", *env_list))

                            # Eye-reflection environment-mapped triangle data
                            if wrote_eye_ref:
                                eye_ref_offset_in_gnc = ef.tell()  # ← Add this
                                env_list = sorted(eye_ref_vertex_indices)
                                ef.write(struct.pack("<I", len(env_list)))
                                ef.write(struct.pack(f"<{len(env_list)}H", *env_list))

                            if wrote_env_mapped:
                                struct.pack_into("<I", model_data, 92, env_offset_in_gnc - 24)

                            if wrote_eye_ref:
                                struct.pack_into("<I", model_data, 96, eye_ref_offset_in_gnc - 24)



                    # Write model section
                    f.write(model_data)


                # Terminator "nextTexture" block (all zeroes)


            self.report({'INFO'}, f"Exported model to {self.filepath}")
            return {'FINISHED'}

        except Exception as e:
            self.report({'ERROR'}, f"Failed to export: {e}")
            return {'CANCELLED'}

    def invoke(self, context, event):
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
                cameraAntic = unpack("<6i", f.read(24))  # 1 + 5
                flags = unpack("<I", f.read(4))[0]
                introID, markupID = unpack("<hh", f.read(4))
                px, py, pz = unpack("<3f", f.read(12))
                bbox = unpack("<6h", f.read(12))
                poly_offset = unpack("<I", f.read(4))[0]
            print(f"[DEBUG] Seeking to polyline offset: {poly_offset + data_start} (poly_offset: {poly_offset})")

            # Save and jump to polyline
            cur = f.tell()
            f.seek(poly_offset + data_start)

            numPoints = unpack("<I", f.read(4))[0]
            f.read(12)  # Skip 12 bytes

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

        # Optional: display as wire and in front
        mesh_obj.display_type = 'WIRE'
        mesh_obj.show_in_front = True



def int_to_flag_set(flag_value):
    return {f"{1<<i:#x}" for i in range(32) if flag_value & (1 << i)}

def import_model_targets(filepath, target_offset, num_targets, num_relocations):
    import os

    section_index = None
    with open(filepath, "rb") as f:
        f.seek(24)  # Start of relocation table
        for _ in range(num_relocations):
            packed = int.from_bytes(f.read(2), "little")
            section_index_candidate = packed >> 3
            f.read(2)  # typeSpecific, skip
            offset = int.from_bytes(f.read(4), "little")

            if offset == 136:
                section_index = section_index_candidate
                break

    if section_index is None:
        print("[INFO] No relocation with offset 136 found.")
        return []

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

        # Create torus at origin
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

        # Rotate 90 degrees on X
        torus.rotation_euler = Euler((radians(90), 0, 0), 'XYZ')
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)

        # Scale up 10x
        torus.scale = (1000.0, 1000.0, 1000.0)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

        # Final transform relative to bone
        # Parent to bone directly
        # Compute bone rest matrix
        bone_mat = armature_obj.matrix_world @ armature_obj.data.bones[bone_name].matrix_local

        # Build local matrix from position and rotation
        local_mat = Matrix.Translation(Vector((px, py, pz))) @ Euler((rx, ry, rz), 'XYZ').to_matrix().to_4x4()

        # World transform = bone rest matrix * local offset
        world_mat = bone_mat @ local_mat

        # Apply local transform relative to "Targets" empty
        torus.matrix_world = target_parent.matrix_world.inverted() @ world_mat

        # Parent to Targets empty
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


        # Assign to bone
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
    # Converts ushort shifted left by 16 bits to float using IEEE 754
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

            # ClothSetup offset from relocation[0]
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

            # Resolve pointers
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

            # ClothJointMaps
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

        # Remove existing Cloth mesh and empty
        for obj in list(bpy.data.objects):
            if obj.name == "Cloth":
                bpy.data.objects.remove(obj, do_unlink=True)

        # Create new Cloth parent empty
        cloth_empty = bpy.data.objects.new("Cloth", None)
        cloth_empty.empty_display_type = 'SPHERE'
        cloth_empty.empty_display_size = 0.5
        bpy.context.collection.objects.link(cloth_empty)
        cloth_empty.parent = armature_obj
        cloth_empty.matrix_parent_inverse.identity()

        # Generate cloth points in world space
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

            # Create UV sphere if flag == 1
            if pt["flags"] == 1:
                bpy.ops.mesh.primitive_uv_sphere_add(radius=20, segments=16, ring_count=8, location=world_pos)
                sphere_obj = bpy.context.active_object
                sphere_obj.name = f"ClothCollision_{i}"
                sphere_obj.parent = cloth_empty
                sphere_obj.matrix_parent_inverse.identity()

                # Set wireframe + in front
                sphere_obj.display_type = 'WIRE'
                sphere_obj.show_in_front = True

                # Add vertex group for bone and assign weight
                vg = sphere_obj.vertex_groups.new(name=bone_name)
                vg.add(range(len(sphere_obj.data.vertices)), 1.0, 'REPLACE')

                # Add armature modifier
                arm_mod = sphere_obj.modifiers.new(name="Armature", type='ARMATURE')
                arm_mod.object = armature_obj

        # Create edges
        edges = []
        for jm in joint_maps:
            a, b = jm["points"][0], jm["points"][1]
            if a < len(verts) and b < len(verts) and a != b:
                edges.append((a, b))
        for rule in dist_rules:
            a, b = rule["point0"], rule["point1"]
            if a < len(verts) and b < len(verts) and a != b:
                edges.append((a, b))

        # Create mesh and object
        mesh = bpy.data.meshes.new("ClothStrip")
        mesh.from_pydata(verts, edges, [])
        mesh.update()

        obj = bpy.data.objects.new("ClothStrip", mesh)
        obj.display_type = 'WIRE'
        obj.show_wire = True
        bpy.context.collection.objects.link(obj)

        # Clear previous data
        if hasattr(obj, "cloth_points"):
            obj.cloth_points.clear()

        for pt in cloth_points:
            item = obj.cloth_points.add()
            item.segment = pt["segment"]
            item.flags = pt["flags"]
            item.joint_order = pt["joint_order"]
            item.up_to = pt["up_to"]
            item.pos = pt["pos"]

        # Assign cloth data to the mesh object
        obj.tr7ae_cloth.gravity = gravity
        obj.tr7ae_cloth.drag = drag
        obj.tr7ae_cloth.wind_response = wind_response
        obj.tr7ae_cloth.flags = flags

        # Parent to the new Cloth empty
        obj.parent = cloth_empty
        obj.matrix_parent_inverse.identity()

        # Skin to bones
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

        # Add armature modifier
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

        # Populate joint maps
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

    print(f"[EnvMap] Loaded {count} env-mapped vertex indices.")
    return indices

def import_eyerefenvmapped_vertices(filepath, eye_ref_env_mapped_vertices, num_relocations):
    import struct, os

    section_index = None
    skip_extra_bytes = False

    # Correct relocation scan
    with open(filepath, "rb") as f:
        f.seek(24)  # 👈 START OF RELOCATIONS
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

    print(f"[EyeRefEnvMap] Loaded {count} eyerefenv-mapped vertex indices.")
    return indices



def import_vertex_colors(filepath, vertex_color_offset, num_relocations, num_vertices):
    import struct, os

    section_index = None

    with open(filepath, "rb") as f:
        f.seek(24)  # Relocations start at offset 24

        for i in range(num_relocations):
            entry_start = f.tell()
            packed = int.from_bytes(f.read(2), "little")  # typeAndSectionInfo
            section_index_candidate = packed >> 3  # upper 13 bits
            f.read(2)  # typeSpecific (ignored)
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
            vertex_colors.append((b / 127.0, g / 127.0, r / 127.0, a / 127.0)) #Could be / 255.0 instead. Actually, that's what ChatGPT initially did. But vertex colors look more accurate this way. We'll see.

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
    "tr7ae_textures"
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
                struct.pack_into('<I', header, 8, 0x0002100F)  # flags
                struct.pack_into('<I', header, 12, height)
                struct.pack_into('<I', header, 16, width)
                struct.pack_into('<I', header, 20, max(1, dxt_size))  # pitch or linear size
                struct.pack_into('<I', header, 28, mipmaps if mipmaps > 0 else 1)

                if dds_format == 21:  # ARGB (uncompressed)
                    struct.pack_into('<I', header, 76, 32)  # size of pixel format
                    struct.pack_into('<I', header, 80, 0x00000041)  # flags: RGBA
                    struct.pack_into('<4s', header, 84, b'\x00\x00\x00\x00')  # no FourCC
                    struct.pack_into('<I', header, 88, 32)  # RGB Bit Count
                    struct.pack_into('<I', header, 92, 0x00FF0000)  # R
                    struct.pack_into('<I', header, 96, 0x0000FF00)  # G
                    struct.pack_into('<I', header, 100, 0x000000FF) # B
                    struct.pack_into('<I', header, 104, 0xFF000000) # A
                else:  # Assume compressed (like DXT1/DXT5)
                    struct.pack_into('<I', header, 76, 32)
                    struct.pack_into('<I', header, 80, 0x00000004)  # flags: FourCC
                    struct.pack_into('<4s', header, 84, struct.pack("<I", dds_format))

                struct.pack_into('<I', header, 108, 0x1000)  # caps
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
        converted_textures = {}  # Avoid converting the same .pcd multiple times
        loaded_images = {}       # Cache for already loaded Blender images
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
            raise ValueError("Invalid file format (missing SECT header)")

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

            current_pos = file.tell()  # Save position
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

                # ── Parser section (where you read num_hboxes and hbox_list) ──
                # … inside your import routine, where you parse hinfo …

                if num_hboxes > 0 and hbox_list != 0:
                    file.seek(hbox_list + section_info_size)
                    boxes = []
                    for _ in range(num_hboxes):
                        # 1) widths
                        wx, wy, wz, ww = struct.unpack("<4f", file.read(16))
                        # 2) positions
                        positionboxx, positionboxy, positionboxz, positionboxw = struct.unpack("<4f", file.read(16))
                        # 3) orientation quaternion
                        qx, qy, qz, qw = struct.unpack("<4f", file.read(16))
                        # 4) integer fields
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
                file.seek(current_pos)  # Restore position

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

            # Parse and attach hInfo structure if applicable
            if do_import_hinfo:
                hinfo_struct = read_hinfo(fhandle, hInfo[0], section_info_size)
                bone_data["hinfo"] = hinfo_struct

            bones.append(bone_data)
            parent_idx = parent[0]
            world_positions[i] = pivot if parent_idx < 0 else world_positions[parent_idx] + pivot

        from mathutils import Matrix

        bone_matrices = {}
        for i, bone in enumerate(bones):
            # Assume identity since we don't have rotations here — just translations
            # But if you ever expand to support rotations, plug them here.
            trans = Matrix.Translation(world_positions[i])
            bone_matrices[i] = trans.to_3x3()  # extract 3x3 for normals

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
            segment = read("<h")[0]  # << segment is read here
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
            texture_id        =  tpageid        & 0x1FFF          # bits  0–12 (13 bits)
            blend_value       = (tpageid >> 13) & 0xF             # bits 13–16 (4 bits)
            unknown_1         = (tpageid >> 17) & 0x7             # bits 17–19 (3 bits)
            unknown_2         = (tpageid >> 20) & 0x1             # bit     20 (1 bit)
            single_sided      = (tpageid >> 21) & 0x1             # bit     21 (1 bit)
            texture_wrap      = (tpageid >> 22) & 0x3             # bits 22–23 (2 bits)
            unknown_3         = (tpageid >> 24) & 0x1             # bit     24 (1 bit)
            unknown_4         = (tpageid >> 25) & 0x1             # bit     25 (1 bit)
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
        # Right after creating your armature_obj:
        armature_obj["max_rad"] = max_rad
        armature_obj["max_rad_sq"] = max_rad_sq
        armature_obj["cdcRenderDataID"] = cdcRenderDataID
        armature_obj.tr7ae_model_scale = scale_vals[:3]
        if bone_mirror_data_offset > 0:
            armature_obj["bone_mirror_data"] = bone_mirror_data
        # Scale down the armature and apply transforms
        armature_obj.scale = (0.01, 0.01, 0.01)
        bpy.context.view_layer.update()
        bpy.context.view_layer.objects.active = armature_obj
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

        from pathlib import Path
        import os

        # Store default main section index (usually 0)
        main_file_index = int(Path(filepath).stem.split("_")[0])
        armature_obj.tr7ae_sections.main_file_index = main_file_index

        # Look for relocation offset 100 (extra data like vertex colors)
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

        # Try to detect a cloth file and its section index
        folder = Path(filepath).parent
        for file in folder.glob("*.cloth"):
            with open(file, "rb") as cf:
                cf.seek(24)
                packed = int.from_bytes(cf.read(2), "little")
                cloth_index = packed >> 3
                armature_obj.tr7ae_sections.cloth_file_index = cloth_index
                print(f"[INFO] Cloth section index = {cloth_index}")
                break

        # Set Armature Display to Stick and enable "Show In Front"
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

                # ── Importer section (where you consume hinfo for each bone) ──
                if "boxes" in hinfo:
                    pbone.tr7ae_hboxes.clear()
                    for box in hinfo["boxes"]:
                        item = pbone.tr7ae_hboxes.add()
                        # full-width
                        item.widthx = box["widthx"]
                        item.widthy = box["widthy"]
                        item.widthz = box["widthz"]
                        item.widthw = box["widthw"]
                        # full-position
                        item.positionboxx   = box["positionboxx"]
                        item.positionboxy   = box["positionboxy"]
                        item.positionboxz   = box["positionboxz"]
                        item.positionboxw   = box["positionboxw"]
                        # orientation
                        item.quatx  = box["quatx"]
                        item.quaty  = box["quaty"]
                        item.quatz  = box["quatz"]
                        item.quatw  = box["quatw"]
                        # flags & IDs
                        item.flags  = box.get("flags", 0)
                        item.id     = box.get("id", 0)
                        item.rank   = box.get("rank", 0)
                        # physical properties
                        item.mass             = box.get("mass", 0)
                        item.buoyancy_factor  = box.get("buoyancy_factor", 0)
                        item.explosion_factor = box.get("explosion_factor", 0)
                        item.material_type    = box.get("material_type", 0)
                        item.pad              = box.get("pad", 0)
                        item.damage           = box.get("damage", 0)
                        item.pad1             = box.get("pad1", 0)

                # ── Import HCapsules into the pose-bone custom props ──
                if "capsules" in hinfo:
                    # clear out any old entries
                    pbone.tr7ae_hcapsules.clear()

                    # for each raw capsule dict in your hinfo…
                    for cap in hinfo["capsules"]:
                        item = pbone.tr7ae_hcapsules.add()

                        # positions (floats)
                        item.posx = cap.get("posx", 0.0)
                        item.posy = cap.get("posy", 0.0)
                        item.posz = cap.get("posz", 0.0)
                        item.posw = cap.get("posw", 1.0)

                        # orientation (quaternion)
                        item.quatx = cap.get("quatx", 0.0)
                        item.quaty = cap.get("quaty", 0.0)
                        item.quatz = cap.get("quatz", 0.0)
                        item.quatw = cap.get("quatw", 1.0)

                        # flags & identifiers
                        item.flags = cap.get("flags", 0)
                        item.id    = cap.get("id",    0)
                        item.rank  = cap.get("rank",  0)

                        # geometry
                        item.radius = cap.get("radius", 0)
                        item.length = cap.get("length", 0)

                        # physical
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

                # Create the HInfo object (parent for all markers)
                if "HInfo" not in bpy.data.objects:
                    hinfo_obj = bpy.data.objects.new("HInfo", None)
                    hinfo_obj.empty_display_type = 'PLAIN_AXES'
                    hinfo_obj.empty_display_size = 0.3
                    context.collection.objects.link(hinfo_obj)

                    # Parent HInfo to Armature for organization
                    hinfo_obj.parent = armature_obj
                    hinfo_obj.matrix_parent_inverse.identity()
                else:
                    hinfo_obj = bpy.data.objects["HInfo"]

                # Get HInfo's collection
                hinfo_collection = hinfo_obj.users_collection[0]

                # Create marker meshes
                if "markers" in hinfo:
                    for marker in hinfo["markers"]:
                        marker_name = f"HMarker_{global_marker_index}"
                        global_marker_index += 1
                        bone_name = f"Bone_{i}"

                        # Create UV sphere mesh
                        bpy.ops.mesh.primitive_cone_add(radius1=0.02, depth=0.05, vertices=8)
                        marker_obj = bpy.context.active_object
                        marker_obj.name = marker_name
                        marker_obj.scale = (100.0, 100.0, 100.0)
                        marker_obj["tr7ae_type"] = "HMarker"
                        marker_obj["tr7ae_bone_index"] = i

                        # Get the bone's rest matrix (local to armature space)
                        bone_matrix = armature_obj.matrix_world @ armature_obj.data.bones[bone_name].matrix_local
                        local_pos = Vector(marker['pos']).to_4d()
                        world_pos = bone_matrix @ local_pos
                        marker_obj.location = hinfo_obj.matrix_world.inverted() @ world_pos.to_3d()


                        rx, ry, rz = marker['rot']

                        marker_obj.rotation_mode = 'ZYX'
                        marker_obj.rotation_euler = (rx, ry, rz)


                        # Parent to HInfo for structure
                        marker_obj.parent = hinfo_obj
                        marker_obj.matrix_parent_inverse.identity()

                        # Add to same collection
                        if marker_obj.name not in hinfo_collection.objects:
                            hinfo_collection.objects.link(marker_obj)

                        # Add vertex group with weight = 1.0 for the bone
                        vg = marker_obj.vertex_groups.new(name=bone_name)
                        bpy.context.view_layer.objects.active = marker_obj
                        bpy.ops.object.mode_set(mode='EDIT')
                        bpy.ops.mesh.select_all(action='SELECT')
                        bpy.ops.object.mode_set(mode='OBJECT')
                        vg.add(range(len(marker_obj.data.vertices)), 1.0, 'REPLACE')

                        # Add Armature Modifier
                        arm_mod = marker_obj.modifiers.new(name="Armature", type='ARMATURE')
                        arm_mod.object = armature_obj

                if "spheres" in hinfo:
                    for j, sphere in enumerate(hinfo["spheres"]):
                        sphere_name = f"HSphere_{global_sphere_index}"
                        global_sphere_index += 1
                        bone_name = f"Bone_{i}"

                        # Create a unit UV-sphere
                        bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, segments=16, ring_count=8)
                        sphere_obj = bpy.context.active_object
                        sphere_obj.name = sphere_name

                        # tag this mesh as an HSphere and record its bone index
                        sphere_obj["tr7ae_type"]       = "HSphere"
                        sphere_obj["tr7ae_bone_index"] = i

                        # Scale to match the imported radius
                        r = sphere["radius"]
                        sphere_obj.scale = (r, r, r)

                        # Show as wireframe and always in front
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

                        # create the cube
                        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 0))
                        box_obj = bpy.context.active_object
                        box_obj.name = box_name

                        # viewport settings
                        box_obj.show_in_front = True
                        box_obj.display_type  = 'WIRE'

                        # 1) Scale to half-extents

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

                        # Convert to world and then to HInfo-local space
                        bone_mat = armature_obj.matrix_world @ armature_obj.data.bones[bone_name].matrix_local
                        world_box_mat = bone_mat @ local_box_mat
                        box_obj.matrix_world = hinfo_obj.matrix_world.inverted() @ world_box_mat

                        # 4) Parent into the HInfo empty and lock in your local transform
                        box_obj.parent                  = hinfo_obj
                        box_obj.parent_type             = 'OBJECT'
                        box_obj.matrix_parent_inverse.identity()

                        # move it into the HInfo collection for neatness
                        if box_obj.name not in hinfo_collection.objects:
                            hinfo_collection.objects.link(box_obj)
                            for coll in box_obj.users_collection:
                                if coll is not hinfo_collection:
                                    coll.objects.unlink(box_obj)

                        box_obj.parent      = hinfo_obj
                        box_obj.parent_type = 'OBJECT'

                        # move it into the HInfo collection for neatness
                        if box_obj.name not in hinfo_collection.objects:
                            hinfo_collection.objects.link(box_obj)
                            for coll in box_obj.users_collection:
                                if coll is not hinfo_collection:
                                    coll.objects.unlink(box_obj)

                        # add armature modifier so it deforms with your rig
                        arm_mod = box_obj.modifiers.new(name="Armature", type='ARMATURE')
                        arm_mod.object = armature_obj

                        # vertex-group for the bone
                        vg = box_obj.vertex_groups.new(name=bone_name)
                        vg.add([v.index for v in box_obj.data.vertices], 1.0, 'REPLACE')

                        # custom props
                        box_obj["tr7ae_type"]       = "HBox"
                        box_obj["tr7ae_bone_index"] = i
                        box_obj["tr7ae_damage"]     = box["damage"]

                if "capsules" in hinfo:
                    for j, cap in enumerate(hinfo["capsules"]):
                        capsule_name = f"HCapsule_{global_capsule_index}"
                        global_capsule_index += 1
                        bone_name    = f"Bone_{i}"

                        # 1) Add a unit cylinder (just like we do a unit sphere for HSpheres)
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
                        capsule_obj["tr7ae_type"]       = "HCapsule"
                        capsule_obj["tr7ae_bone_index"] = i

                        # 2) Scale it to the real radius & length
                        radius = cap.get("radius", 1.0)
                        length = cap.get("length", 1.0)
                        # Note: cylinder depth = 1.0 ⇒ Z-scale = length
                        capsule_obj.scale = (radius, radius, length)

                        # 3) Wireframe + always-in-front (just like HSphere)
                        capsule_obj.display_type = 'WIRE'
                        capsule_obj.show_in_front = True

                        # 4) Compute bone-local → world → HInfo-local position
                        pos_vec = Vector((
                            cap.get("posx", 0.0),
                            cap.get("posy", 0.0),
                            cap.get("posz", 0.0),
                            cap.get("posw", 1.0)
                        ))
                        bone_mat = (
                            armature_obj.matrix_world @
                            armature_obj.data.bones[bone_name].matrix_local
                        )
                        world_pos = bone_mat @ pos_vec
                        capsule_obj.location = (
                            hinfo_obj.matrix_world.inverted() @
                            world_pos.to_3d()
                        )

                        # 5) Apply quaternion rotation
                        capsule_obj.rotation_mode = 'QUATERNION'
                        capsule_obj.rotation_quaternion = Quaternion((
                            cap.get("quatw", 1.0),
                            cap.get("quatx", 0.0),
                            cap.get("quaty", 0.0),
                            cap.get("quatz", 0.0),
                        ))

                        # 6) Parent to the HInfo empty + reset parent matrix
                        capsule_obj.parent = hinfo_obj
                        capsule_obj.matrix_parent_inverse.identity()

                        # 7) Vert-group + Armature modifier
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

            if mface_faces:
                # 1. Find all unique vertex indices used in mface_faces
                used_verts = set()
                for tri in mface_faces:
                    used_verts.update(tri)

                # 2. Remap them to new compact indices
                used_verts_sorted = sorted(used_verts)
                index_remap = {old_idx: new_idx for new_idx, old_idx in enumerate(used_verts_sorted)}

                # 3. Rebuild face list with remapped indices
                remapped_faces = [tuple(index_remap[v] for v in face) for face in mface_faces]

                # 4. Filter verts, uvs, etc.
                filtered_verts = [verts[v] for v in used_verts_sorted]
                filtered_uvs = [uvs[v] for v in used_verts_sorted]
                filtered_segments = [segments[v] for v in used_verts_sorted]

                # 5. Create mesh from only used verts
                mface_mesh = bpy.data.meshes.new("TR7AE_MFace_Mesh")
                # Full vertex list
                mface_mesh.from_pydata(verts, [], mface_faces)


                import random

                # Generate a map of group_id -> color
                group_color_map = {}
                def get_group_color(group_id):
                    if group_id not in group_color_map:
                        random.seed(group_id)  # Deterministic color per group
                        group_color_map[group_id] = (
                            random.random(), random.random(), random.random(), 1.0
                        )
                    return group_color_map[group_id]

                # New vertex color layer
                group_vcol_layer = mface_mesh.vertex_colors.new(name="DryingGroups")

                for poly in mface_mesh.polygons:
                    same = mface_samebits[poly.index]
                    gid0 = (same >> 0) & 0b11111
                    gid1 = (same >> 5) & 0b11111
                    gid2 = (same >> 10) & 0b11111

                    group_colors = [
                        get_group_color(gid0),
                        get_group_color(gid1),
                        get_group_color(gid2)
                    ]

                    for i, loop_idx in enumerate(poly.loop_indices):
                        group_vcol_layer.data[loop_idx].color = group_colors[i]




                if uvs:
                    mface_mesh.uv_layers.new(name="UVMap")
                    uv_layer = mface_mesh.uv_layers.active.data
                    for poly in mface_mesh.polygons:
                        for loop_index in range(poly.loop_start, poly.loop_start + poly.loop_total):
                            v_idx = mface_mesh.loops[loop_index].vertex_index
                            uv_layer[loop_index].uv = uvs[v_idx]

                mface_obj = bpy.data.objects.new("MFace_Mesh", mface_mesh)
                context.collection.objects.link(mface_obj)
                mface_obj.parent = armature_obj
                mface_obj["tr7ae_is_mface"] = True

                # Armature modifier and bone weights
                mod = mface_obj.modifiers.new(name="Armature", type='ARMATURE')
                mod.object = armature_obj

                for i in range(len(bones)):
                    mface_obj.vertex_groups.new(name=f"Bone_{i}")

                for i, seg in enumerate(segments):
                    vg = mface_obj.vertex_groups.get(f"Bone_{seg}")
                    if vg:
                        vg.add([i], 1.0, 'REPLACE')

                for first_vertex, last_vertex, index, weight_index, weight in virtsegment_data:
                    vg_primary = mface_obj.vertex_groups.get(f"Bone_{index}")
                    vg_secondary = mface_obj.vertex_groups.get(f"Bone_{weight_index}")
                    if vg_primary and vg_secondary:
                        for v in range(first_vertex, last_vertex + 1):
                            if v < len(verts):
                                vg_secondary.add([v], weight, 'REPLACE')
                                vg_primary.add([v], 1.0 - weight, 'ADD')


        for mesh_index, (chunk_verts, chunk_faces, chunk_uvs, chunk_normals, chunk_segments, draw_group, chunk_vert_map, texture_id, blend_value, unknown_1, unknown_2, single_sided, texture_wrap, unknown_3, unknown_4, flat_shading, sort_z, stencil_pass, stencil_func, alpha_ref) in enumerate(mesh_chunks):
            mesh = bpy.data.meshes.new(f"TR7AE_Mesh_{mesh_index}")
            mesh.from_pydata(chunk_verts, [], chunk_faces)
            mesh.update()
            if vertex_colors:
                color_layer = mesh.vertex_colors.new(name="Color")
                color_data = color_layer.data
                for poly in mesh.polygons:
                    for loop_idx in poly.loop_indices:
                        vertex_idx = mesh.loops[loop_idx].vertex_index
                        original_idx = list(chunk_vert_map.keys())[list(chunk_vert_map.values()).index(vertex_idx)]
                        color_data[loop_idx].color = vertex_colors[original_idx]
            # Determine if any original/global vertex used in this chunk is in envMapped list
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

            # Clear default nodes
            for node in nodes:
                nodes.remove(node)

            # Create new nodes
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
                hex_id = f"{texture_id:x}"  # lowercase hex
                pattern = os.path.join(folder, f"*_{hex_id}.pcd")
                matches = glob.glob(pattern)
                return matches[0] if matches else None
            
            folder = os.path.dirname(filepath)
            pcd_path = find_texture_by_id(folder, texture_id)

            # Place nodes first
            output.location = (300, 0)
            bsdf.location = (0, 0)
            tex_image.location = (-600, 0)

            # Link BSDF to Output
            links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])

            # Now load and connect texture
            from pathlib import Path

            pcd_path = find_texture_by_id(folder, texture_id)

            # Convert or reuse DDS texture
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

            # Always get or load the image, even if already converted
                if str(dds_path) in loaded_images:
                    image = loaded_images[str(dds_path)]
                elif dds_path.exists():
                    image = bpy.data.images.load(str(dds_path))
                    loaded_images[str(dds_path)] = image
                    image.pack()
                else:
                    image = None  # Texture missing

                # Always assign the image if it exists
                if image:
                    tex_image.image = image
                else:
                    print(f"[ERROR] DDS file not found or not written: {dds_path}")

            # Vertex Color AO via Multiply
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

                    # Output node
                    output = nodes.new(type='ShaderNodeOutputMaterial')
                    output.location = (1000, 0)

                    # Principled BSDF
                    principled = nodes.new(type='ShaderNodeBsdfPrincipled')
                    principled.location = (600, 0)
                    principled.inputs['Roughness'].default_value = 0.5

                    # Multiply for Alpha Control
                    multiply_alpha = nodes.new(type='ShaderNodeMixRGB')
                    multiply_alpha.blend_type = 'MULTIPLY'
                    multiply_alpha.inputs['Fac'].default_value = 1.0
                    multiply_alpha.location = (200, -200)

                    # Multiply for Color Boost
                    multiply_boost = nodes.new(type='ShaderNodeMixRGB')
                    multiply_boost.blend_type = 'MULTIPLY'
                    multiply_boost.inputs['Fac'].default_value = 1.0
                    multiply_boost.inputs['Color2'].default_value = (5.0, 5.0, 5.0, 1.0)
                    multiply_boost.location = (200, 100)

                    # Texture Image
                    tex_image = nodes.new(type='ShaderNodeTexImage')
                    tex_image.image = image
                    tex_image.interpolation = 'Linear'
                    tex_image.extension = 'EXTEND'
                    tex_image.projection = 'BOX'
                    tex_image.projection_blend = 1.0
                    tex_image.location = (-400, 100)

                    # Attribute Node (for Color Alpha)
                    vc_node = nodes.new(type='ShaderNodeAttribute')
                    vc_node.attribute_name = "Color"
                    vc_node.location = (-400, -100)

                    # Mapping
                    mapping = nodes.new(type='ShaderNodeMapping')
                    mapping.vector_type = 'TEXTURE'
                    mapping.location = (-800, 0)
                    mapping.inputs['Location'].default_value = (-2.0, 0.0, -2.0)
                    mapping.inputs['Rotation'].default_value = (0.0, 0.0, 0.0)
                    mapping.inputs['Scale'].default_value = (4.0, 4.0, 4.0)

                    # Texture Coordinate
                    tex_coord = nodes.new(type='ShaderNodeTexCoord')
                    tex_coord.location = (-1000, 0)

                    # ---- Links ----

                    # Texture Coord → Mapping
                    links.new(tex_coord.outputs['Reflection'], mapping.inputs['Vector'])
                    # Mapping → Texture
                    links.new(mapping.outputs['Vector'], tex_image.inputs['Vector'])

                    # Image Color → Multiply Boost Color1 (A)
                    links.new(tex_image.outputs['Color'], multiply_boost.inputs['Color1'])
                    # Multiply Boost → Principled Base Color
                    links.new(multiply_boost.outputs['Color'], principled.inputs['Base Color'])

                    # Image Color → Multiply Alpha Color1 (A)
                    links.new(tex_image.outputs['Color'], multiply_alpha.inputs['Color1'])
                    # Attribute Alpha → Multiply Alpha Color2 (B)
                    links.new(vc_node.outputs['Alpha'], multiply_alpha.inputs['Color2'])
                    # Multiply Alpha → Principled Alpha
                    links.new(multiply_alpha.outputs['Color'], principled.inputs['Alpha'])

                    # Principled → Output
                    links.new(principled.outputs['BSDF'], output.inputs['Surface'])


                if blend_value == 1:
                    mat.use_nodes = True
                    mat.blend_method = 'BLEND'

                    nodes = mat.node_tree.nodes
                    links = mat.node_tree.links
                    nodes.clear()

                    # Output node
                    output = nodes.new(type='ShaderNodeOutputMaterial')
                    output.location = (800, 0)

                    # Principled BSDF
                    principled = nodes.new(type='ShaderNodeBsdfPrincipled')
                    principled.location = (400, 0)
                    principled.inputs['Roughness'].default_value = 0.5

                    # Texture Image
                    tex_image = nodes.new(type='ShaderNodeTexImage')
                    tex_image.image = image
                    tex_image.interpolation = 'Linear'
                    tex_image.extension = 'EXTEND'
                    tex_image.projection = 'BOX'
                    tex_image.projection_blend = 1.0
                    tex_image.location = (-400, 100)

                    # Attribute Node (for Color Alpha)
                    vc_node = nodes.new(type='ShaderNodeAttribute')
                    vc_node.attribute_name = "Color"
                    vc_node.location = (-400, -100)

                    # Mapping
                    mapping = nodes.new(type='ShaderNodeMapping')
                    mapping.vector_type = 'TEXTURE'
                    mapping.location = (-800, 0)
                    mapping.inputs['Location'].default_value = (-2.0, 0.0, -2.0)
                    mapping.inputs['Rotation'].default_value = (0.0, 0.0, 0.0)
                    mapping.inputs['Scale'].default_value = (4.0, 4.0, 4.0)

                    # Texture Coordinate
                    tex_coord = nodes.new(type='ShaderNodeTexCoord')
                    tex_coord.location = (-1000, 0)

                    # Links
                    links.new(tex_coord.outputs['Reflection'], mapping.inputs['Vector'])
                    links.new(mapping.outputs['Vector'], tex_image.inputs['Vector'])
                    links.new(tex_image.outputs['Color'], principled.inputs['Base Color'])
                    links.new(vc_node.outputs['Alpha'], principled.inputs['Alpha'])
                    links.new(principled.outputs['BSDF'], output.inputs['Surface'])

            # Set officially registered props (avoids Custom Properties tab)
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

class TR7AE_OT_NormalizeAndLimitWeights(bpy.types.Operator):
    bl_idname = "tr7ae.normalize_and_limit_weights"
    bl_label = "Limit Weights and Normalize All"
    bl_description = "Automatically limit all weights per vertex to 2 and normalize them"

    def execute(self, context):
        obj = context.active_object

        # Armature: all mesh children; Mesh: only self
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
                limit=0.0001,  # very low to catch near-zero weights
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
    bl_category = 'TR7AE'
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        if not obj or obj.type not in {'MESH', 'ARMATURE'}:
            return False

        # Skip specific tagged types
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

        # Determine whether to hide or show based on first HSphere's current state
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

        # Determine whether to hide or show based on first HSphere's current state
        should_hide = not hcapsules[0].hide_get()

        for obj in hcapsules:
            obj.hide_set(should_hide)

        action = "Hidden" if should_hide else "Shown"
        self.report({'INFO'}, f"{action} {len(hcapsules)} HCapsules(s)")
        return {'FINISHED'}

class TR7AE_PT_Tools(Panel):
    bl_label = "Tomb Raider 7/AE Mesh Editor"
    bl_idname = "TR7AE_PT_Tools"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'TR7AE'

    _sphere_expand = {}
    _box_expand = {}
    _marker_expand = {}
    _capsule_expand = {}

    def draw(self, context):
        layout = self.layout
        layout.operator(TR7AE_OT_ImportModel.bl_idname, icon='IMPORT')

        if context.active_object and context.active_object.type == 'ARMATURE':
            layout.operator("tr7ae.export_custom_model", icon='EXPORT')

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
                    # Use texconv to convert to DDS
                    temp_dds = Path(tempfile.gettempdir()) / (full_path.stem + ".dds")
                    prefs = context.preferences.addons[__name__].preferences
                    texconv_path = getattr(prefs, 'texconv_path', '') or "texconv.exe"
                    format = "DXT1" if is_image_fully_opaque_blender(full_path) else "DXT5"

                    result = subprocess.run([
                        texconv_path,
                        "-nologo",
                        "-y",
                        "-ft", "dds",
                        "-f", format,
                        "-o", str(temp_dds.parent),
                        str(full_path)
                    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                    if result.returncode != 0 or not temp_dds.exists():
                        stderr = result.stderr.decode().strip()
                        stdout = result.stdout.decode().strip()
                        raise RuntimeError(f"texconv failed:\n{stderr or '(no stderr output)'}\n{stdout or ''}")

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
    
class TR7AE_OT_ConvertPCDToDDS(Operator):
    bl_idname = "tr7ae.convert_pcd_to_dds"
    bl_label = "Convert PCD to DDS"
    bl_description = "Convert PCD texture files to DDS format"

    files: CollectionProperty(type=OperatorFileListElement)
    directory: StringProperty(subtype='DIR_PATH')
    filepath: StringProperty(subtype='FILE_PATH')
    filter_glob: StringProperty(default="*.pcd", options={'HIDDEN'})

    def execute(self, context):
        try:
            targets = self.files or [Path(self.filepath).name]

            for f in targets:
                full_path = Path(self.directory) / f.name
                if not full_path.exists():
                    continue
                output_directory = full_path.parent
                convert_pcd_to_dds(full_path, output_directory)

            self.report({'INFO'}, "PCD to DDS conversion complete.")
            return {'FINISHED'}
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}

    def invoke(self, context, event):
        self.filter_glob = "*.pcd"
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

# Add button to UI panel
class TR7AE_PT_TextureTools(bpy.types.Panel):
    bl_label = "Texture Tools"
    bl_idname = "TR7AE_PT_TextureTools"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'TR7AE'

    def draw(self, context):
        layout = self.layout
        layout.operator("tr7ae.convert_image_to_pcd", text="Convert Image(s) to PCD")
        layout.operator("tr7ae.convert_pcd_to_dds", text="Convert PCD to DDS")

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
        
class TR7AE_PT_FileSectionsPanel(bpy.types.Panel):
    bl_label = "File Section Data"
    bl_idname = "TR7AE_PT_file_sections"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'TR7AE'
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
    bl_category = 'Tomb Raider 7/AE Mesh Editor'
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
    bl_category = 'TR7AE'
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

        layout.label(text="Radius Properties:")
        if "max_rad" in obj:
            layout.prop(obj, '["max_rad"]', text="Max Radius")
        if "max_rad_sq" in obj:
            layout.prop(obj, '["max_rad_sq"]', text="Max Radius Squared")
        if "cdcRenderDataID" in obj:
            layout.prop(obj, '["cdcRenderDataID"]', text="cdcRenderDataID")
        layout.label(text="Model Scale:")
        layout.prop(obj, "tr7ae_model_scale", text="")

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

# UI panel to display/edit selected vertex's ClothPoint
class TR7AE_PT_ClothPointInspector(bpy.types.Panel):
    bl_label = "ClothPoint"
    bl_idname = "TR7AE_PT_clothpoint_inspector"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'TR7AE'

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
    bl_category = 'TR7AE'

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
    bl_category = 'TR7AE'

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
    bl_label = "Mesh Info"
    bl_idname = "TR7AE_PT_DrawGroupPanel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'TR7AE'

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
    bl_category = 'TR7AE'
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
    bl_category = 'TR7AE'
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.type == 'MESH' and obj.get("tr7ae_type") == "HSphere"

    def draw(self, context):
        layout = self.layout
        obj = context.active_object

        # must have a bone index
        bone_index = obj.get("tr7ae_bone_index")
        if bone_index is None:
            layout.label(text="Missing tr7ae_bone_index")
            return

        # find the armature and pose bone
        arm = obj.find_armature()
        if not arm:
            layout.label(text="No armature found.")
            return
        bone = arm.pose.bones.get(f"Bone_{bone_index}")
        if not bone or not bone.tr7ae_hspheres:
            layout.label(text=f"No HSpheres for Bone_{bone_index}")
            return

        # parse sphere index from name: HSphere_<bone>_<idx>
        try:
            global_index = int(obj.name.split("_")[-1])
        except:
            layout.label(text="Invalid object name.")
            return

        # Search globally through all tr7ae_hspheres
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

        # Draw the sphere data fields
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

        # Add a new data entry to the bone
        sphere = bone.tr7ae_hspheres.add()
        sphere.radius = 10.0
        sphere.x = 0.0
        sphere.y = 0.0
        sphere.z = 0.0
        sphere.id = 0
        sphere.rank = len(bone.tr7ae_hspheres) - 1

        # --- Compute next available global index ---
        existing_indices = [
            int(obj.name.split("_")[1])
            for obj in bpy.data.objects
            if obj.name.startswith("HSphere_") and obj.name.split("_")[1].isdigit()
        ]
        next_index = max(existing_indices, default=-1) + 1

        # --- Create visual mesh ---
        mesh = bpy.data.meshes.new(f"HSphereMesh_{next_index}")
        sphere_obj = bpy.data.objects.new(f"HSphere_{next_index}", mesh)
        context.collection.objects.link(sphere_obj)

        # Set display properties
        sphere_obj.display_type = 'WIRE'
        sphere_obj.show_in_front = True

        # Create a UV sphere using bmesh
        bm = bmesh.new()
        bmesh.ops.create_uvsphere(bm, u_segments=16, v_segments=8, radius=1.0)
        bm.to_mesh(mesh)
        bm.free()

        # Set transform (at origin, scaled to radius)
        bone_mat = arm.matrix_world @ arm.data.bones[bone.name].matrix_local
        sphere_obj.location = bone_mat.to_translation() * 100
        sphere_obj.scale = (sphere.radius, sphere.radius, sphere.radius)

        # Assign custom properties
        sphere_obj["tr7ae_type"] = "HSphere"
        sphere_obj["tr7ae_bone_index"] = list(arm.pose.bones).index(bone)

        # Parent to HInfo empty if it exists
        hinfo_empty = bpy.data.objects.get("HInfo")
        if hinfo_empty:
            sphere_obj.parent = hinfo_empty

        # Add armature modifier to follow armature
        mod = sphere_obj.modifiers.new(name="Armature", type='ARMATURE')
        mod.object = arm

        # Create vertex group for the bone and assign all verts
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

        # --- Add new HMarker data block ---
        marker = bone.tr7ae_hmarkers.add()
        marker.px = 0.0
        marker.py = 0.0
        marker.pz = 0.0
        marker.rx = 0.0
        marker.ry = 0.0
        marker.rz = 0.0
        marker.id = 0
        marker.rank = len(bone.tr7ae_hmarkers) - 1  # local to bone
        marker.bone = bone_index  # stored as int

        # --- Determine globally unique marker.index ---
        existing_indices = [
            m.index
            for pbone in arm.pose.bones
            if hasattr(pbone, "tr7ae_hmarkers")
            for m in pbone.tr7ae_hmarkers
            if hasattr(m, "index")
        ]
        marker.index = max(existing_indices, default=-1) + 1

        # --- Determine next global object name index ---
        existing_obj_indices = [
            int(obj.name.split("_")[1])
            for obj in bpy.data.objects
            if obj.name.startswith("HMarker_") and obj.name.split("_")[1].isdigit()
        ]
        next_obj_index = max(existing_obj_indices, default=-1) + 1
        marker_name = f"HMarker_{next_obj_index}"

        # --- Create cone mesh object ---
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

        # --- Place at bone head ---
        bone_mat = arm.matrix_world @ arm.data.bones[bone.name].matrix_local
        marker_obj.location = bone_mat.to_translation() * 100  # If using cm scale

        # --- Set custom props ---
        marker_obj["tr7ae_type"] = "HMarker"
        marker_obj["tr7ae_bone_index"] = bone_index

        # --- Parent to HInfo empty ---
        hinfo_empty = bpy.data.objects.get("HInfo")
        if hinfo_empty:
            marker_obj.parent = hinfo_empty

        # --- Add armature modifier ---
        mod = marker_obj.modifiers.new(name="Armature", type='ARMATURE')
        mod.object = arm

        # --- Add vertex group for bone ---
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

        # Add a new HBox entry to the bone
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

        # Compute next available global index
        existing_indices = [
            int(obj.name.split("_")[1])
            for obj in bpy.data.objects
            if obj.name.startswith("HBox_") and obj.name.split("_")[1].isdigit()
        ]
        next_index = max(existing_indices, default=-1) + 1

        # Create cube mesh
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 0))
        box_obj = bpy.context.active_object
        box_obj.name = f"HBox_{next_index}"
        box_obj.display_type = 'WIRE'
        box_obj.show_in_front = True

        # Set transform to bone head
        bone_mat = arm.matrix_world @ arm.data.bones[bone.name].matrix_local
        box_obj.location = bone_mat.to_translation() * 100

        # Set scale to half-extents
        box_obj.scale = (hbox.widthx, hbox.widthy, hbox.widthz)

        # Set custom props
        box_obj["tr7ae_type"] = "HBox"
        box_obj["tr7ae_bone_index"] = list(arm.pose.bones).index(bone)

        # Parent to HInfo empty if it exists
        hinfo_empty = bpy.data.objects.get("HInfo")
        if hinfo_empty:
            box_obj.parent = hinfo_empty
            box_obj.matrix_parent_inverse.identity()

        # Add Armature modifier
        mod = box_obj.modifiers.new(name="Armature", type='ARMATURE')
        mod.object = arm

        # Create and assign vertex group
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

        # Add a new HCapsule entry
        hcap = bone.tr7ae_hcapsules.add()
        hcap.radius = 2.0
        hcap.ymin = -5.0
        hcap.ymax = 5.0
        hcap.id = 0
        hcap.rank = len(bone.tr7ae_hcapsules) - 1

        # Compute next global index
        existing_indices = [
            int(obj.name.split("_")[1])
            for obj in bpy.data.objects
            if obj.name.startswith("HCapsule_") and obj.name.split("_")[1].isdigit()
        ]
        next_index = max(existing_indices, default=-1) + 1
        name = f"HCapsule_{next_index}"

        # Create capsule-like mesh using cylinder + bmesh domes
        mesh = bpy.data.meshes.new(f"HCapsuleMesh_{next_index}")
        obj = bpy.data.objects.new(name, mesh)
        context.collection.objects.link(obj)

        bm = bmesh.new()

        # Capsule body
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

        # Position at bone head
        bone_mat = arm.matrix_world @ arm.data.bones[bone.name].matrix_local
        obj.location = bone_mat.to_translation() * 100

        # Set display
        obj.display_type = 'WIRE'
        obj.show_in_front = True
        obj.name = name

        # Assign custom props
        obj["tr7ae_type"] = "HCapsule"
        obj["tr7ae_bone_index"] = list(arm.pose.bones).index(bone)

        # Parent to HInfo empty if available
        hinfo_empty = bpy.data.objects.get("HInfo")
        if hinfo_empty:
            obj.parent = hinfo_empty
            obj.matrix_parent_inverse.identity()

        # Add Armature modifier
        mod = obj.modifiers.new(name="Armature", type='ARMATURE')
        mod.object = arm

        # Create and assign vertex group
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
    bl_category = 'TR7AE'
    bl_options = {'DEFAULT_CLOSED'}

    _hbox_expand = {}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        # show if we're in pose‐mode on an armature bone with HBoxes
        if obj and obj.type == 'ARMATURE' and context.active_pose_bone:
            return bool(context.active_pose_bone.tr7ae_hboxes)
        # or if we have one of the HBox mesh objects selected
        if obj and obj.type == 'MESH' and obj.get("tr7ae_type") == "HBox":
            return True
        return False

    def draw(self, context):
        layout = self.layout
        obj = context.active_object

        # 1) Mesh selected: show global index-based single box
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

        # 2) Armature + Pose Bone context: show all boxes on the active bone
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

        # fallback
        layout.label(text="No HBox data available.")

class TR7AE_PT_AddHInfoButtons(bpy.types.Panel):
    bl_label = "Add HInfo Component"
    bl_idname = "TR7AE_PT_AddHInfoButtons"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'TR7AE'
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
    bl_category = 'TR7AE'
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
    bl_category = 'TR7AE'
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.type == 'MESH' and obj.get("tr7ae_type") == "HCapsule"

    def draw(self, context):
        layout = self.layout
        obj = context.active_object

        # 1) Get bone index
        bone_index = obj.get("tr7ae_bone_index")
        if bone_index is None:
            layout.label(text="Missing tr7ae_bone_index")
            return

        # 2) Find armature & pose‐bone
        arm = obj.find_armature()
        if not arm:
            layout.label(text="No armature found.")
            return
        bone = arm.pose.bones.get(f"Bone_{bone_index}")
        if not bone or not bone.tr7ae_hcapsules:
            layout.label(text=f"No HCapsules for Bone_{bone_index}")
            return

        # 3) Parse capsule index from name: HCapsule_<bone>_<idx>
        try:
            cap_idx = int(obj.name.split("_")[-1])
        except:
            cap_idx = 0
        cap = bone.tr7ae_hcapsules[cap_idx]

        # 4) Draw exactly the same props as in the HCapsule bone‐panel
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
    bl_category = 'TR7AE'
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
    bl_category = 'TR7AE'
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return (
            obj and 
            obj.type == 'MESH' and 
            "HMarker" in obj.name  # fallback check for name
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

        # Extract marker index from object name, e.g. HMarker_3_2 → 2
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

        # Match by global order of all markers
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

        # Enter pose mode
        bpy.ops.object.mode_set(mode='OBJECT')
        context.view_layer.objects.active = armature
        bpy.ops.object.mode_set(mode='POSE')

        # Get target matrix
        target_matrix = hmarker.matrix_world.copy()

        # Convert to armature space
        relative_matrix = armature.matrix_world.inverted() @ target_matrix

        # Remove scale
        location = relative_matrix.to_translation()
        rotation = relative_matrix.to_euler()
        clean_matrix = mathutils.Euler(rotation).to_matrix().to_4x4()
        clean_matrix.translation = location

        # Apply to bone
        first_bone.matrix_basis = clean_matrix

        # Exit pose mode
        bpy.ops.object.mode_set(mode='OBJECT')

        # Parent the armature to the HMarker
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
        # Find all HInfo objects
        hinfo_objects = [
            obj for obj in bpy.data.objects
            if obj.name.startswith(("HSphere_", "HBox_", "HCapsule_", "HMarker_"))
        ]

        if not hinfo_objects:
            self.report({'WARNING'}, "No HInfo objects found.")
            return {'CANCELLED'}

        # Determine target visibility by inspecting current state
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
    bl_category = 'TR7AE'
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
    bl_label = "Material Info"
    bl_idname = "TR7AE_PT_MaterialPanel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'TR7AE'

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

def on_blend_value_changed(self, context):
    mat = self
    if not mat.use_nodes:
        return

    if getattr(mat, 'tr7ae_is_envmapped', False) or getattr(mat, 'tr7ae_is_eyerefenvmapped', False):
        # EBUILD ENV-MAP MATERIAL NODES HERE
        # You can refactor your envmap shader node setup into a helper function
        # like build_envmap_shader(mat, image)
        return

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Get old image before clearing nodes
    old_image = None
    for node in nodes:
        if node.type == 'TEX_IMAGE' and node.image:
            old_image = node.image
            break

    # Fix: Restore original image if old_image is a _NonColor copy
    if old_image and "_NonColor" in old_image.name:
        base_name = old_image.name.replace("_NonColor", "")
        if base_name in bpy.data.images:
            old_image = bpy.data.images[base_name]

    nodes.clear()

    # Nodes: Shared
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

    # Restore original image if it's a _NonColor copy
    if old_image and "_NonColor" in old_image.name:
        base_name = old_image.name.replace("_NonColor", "")
        if base_name in bpy.data.images:
            old_image = bpy.data.images[base_name]

    # Restore original image if it's a _NonColor copy
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

        # Output and BSDF
        output = nodes.new(type='ShaderNodeOutputMaterial')
        output.location = (400, 0)

        bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
        bsdf.location = (200, 0)
        bsdf.inputs['Specular IOR Level'].default_value = 0.0
        bsdf.inputs['Roughness'].default_value = 0.5

        # Texture node
        tex_image = nodes.new(type='ShaderNodeTexImage')
        tex_image.location = (-600, 100)
        if old_image:
            tex_image.image = old_image
        tex_image.image.colorspace_settings.name = "sRGB"

        # Vertex Color
        attr = nodes.new(type='ShaderNodeAttribute')
        attr.attribute_name = "Color"
        attr.location = (-600, -100)

        # Multiply
        mult_node = nodes.new(type='ShaderNodeMixRGB')
        mult_node.blend_type = 'MULTIPLY'
        mult_node.inputs['Fac'].default_value = 1.0
        mult_node.location = (-300, 0)

        # Greater Than (ONLY connected to texture alpha)
        greater = nodes.new(type='ShaderNodeMath')
        greater.operation = 'GREATER_THAN'
        greater.inputs[1].default_value = 0.0
        greater.location = (-300, -200)

        # Connect nodes
        links.new(tex_image.outputs['Color'], mult_node.inputs['Color1'])
        links.new(attr.outputs['Color'], mult_node.inputs['Color2'])

        links.new(tex_image.outputs['Alpha'], greater.inputs[0])

        links.new(mult_node.outputs['Color'], bsdf.inputs['Base Color'])
        links.new(greater.outputs[0], bsdf.inputs['Alpha'])
        links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])

        # Material settings
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
        # Restore original image if it was a Non-Color duplicate
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

        # Texture
        tex_image = nodes.new(type='ShaderNodeTexImage')
        tex_image.location = (-600, 0)
        if old_image:
            tex_image.image = old_image

        # Multiply
        multiply = nodes.new(type='ShaderNodeMath')
        multiply.operation = 'MULTIPLY'
        multiply.inputs[1].default_value = 1.0
        multiply.location = (-400, 0)

        # Glossy
        glossy = nodes.new(type='ShaderNodeBsdfGlossy')
        glossy.inputs['Roughness'].default_value = 0.3
        glossy.location = (-200, 0)

        # Transparent
        transparent = nodes.new(type='ShaderNodeBsdfTransparent')
        transparent.location = (-200, -200)

        # Mix Shader
        mix_shader = nodes.new(type='ShaderNodeMixShader')
        mix_shader.location = (0, 0)

        # Mix value
        mix_value = nodes.new(type='ShaderNodeValue')
        mix_value.outputs[0].default_value = 0.1
        mix_value.location = (-400, -200)

        # Output
        output = nodes.new(type='ShaderNodeOutputMaterial')
        output.location = (200, 0)

        # Connect
        links.new(tex_image.outputs['Alpha'], multiply.inputs[0])
        links.new(multiply.outputs[0], glossy.inputs['Color'])

        links.new(glossy.outputs['BSDF'], mix_shader.inputs[2])
        links.new(transparent.outputs['BSDF'], mix_shader.inputs[1])
        links.new(mix_value.outputs[0], mix_shader.inputs['Fac'])
        links.new(mix_shader.outputs['Shader'], output.inputs['Surface'])

    elif mat.tr7ae_blend_value in (2, 5, 8):
        nodes.clear()
        # Duplicate image to prevent affecting shared usage
        if old_image:
            base_name = old_image.name
            if "_NonColor" in base_name:
                base_name = base_name.replace("_NonColor", "")
                original = bpy.data.images.get(base_name)
                if original:
                    old_image = original

            image_copy_name = old_image.name + "_NonColor"

            # Check if the copy already exists
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

        # Connect image to add color
        links.new(tex_image.outputs['Color'], add_color.inputs['Color1'])

        # Connect add color result to transparent color input
        links.new(add_color.outputs['Color'], transparent.inputs['Color'])

        # Connect same transparent BSDF to both shader inputs
        links.new(transparent.outputs['BSDF'], add_shader.inputs[0])
        links.new(transparent.outputs['BSDF'], add_shader.inputs[1])

        # Output
        links.new(add_shader.outputs['Shader'], output.inputs['Surface'])

        mat.blend_method = 'BLEND'

    elif mat.tr7ae_blend_value == 3:
        nodes.clear()
        output = nodes.new(type='ShaderNodeOutputMaterial')
        output.location = (400, 0)

        # Restore original image if needed
        if old_image and "_NonColor" in old_image.name:
            base_name = old_image.name.replace("_NonColor", "")
            if base_name in bpy.data.images:
                old_image = bpy.data.images[base_name]

        # Texture node
        tex_image = nodes.new(type='ShaderNodeTexImage')
        tex_image.location = (-600, 0)
        if old_image:
            tex_image.image = old_image
        tex_image.image.colorspace_settings.name = "sRGB"

        # Multiply color
        multiply = nodes.new(type='ShaderNodeMixRGB')
        multiply.blend_type = 'MULTIPLY'
        multiply.inputs['Fac'].default_value = 0.5
        multiply.inputs['Color1'].default_value = (1.0, 1.0, 1.0, 1.0)  # use darker color for effect if needed
        multiply.location = (-400, 0)
        links.new(tex_image.outputs['Color'], multiply.inputs['Color2'])

        # Transparent BSDF
        transparent = nodes.new(type='ShaderNodeBsdfTransparent')
        transparent.location = (-200, 0)
        links.new(multiply.outputs['Color'], transparent.inputs['Color'])

        # Output
        links.new(transparent.outputs['BSDF'], output.inputs['Surface'])

        # Material transparency settings
        mat.blend_method = 'BLEND'

    elif mat.tr7ae_blend_value == 4:
        nodes.clear()

        output = nodes.new(type='ShaderNodeOutputMaterial')
        output.location = (400, 0)

        # Preserve texture image even if not connected
        tex_image = nodes.new(type='ShaderNodeTexImage')
        tex_image.location = (-600, 0)
        if old_image:
            tex_image.image = old_image
        tex_image.image.colorspace_settings.name = "sRGB"

        # Principled BSDF (Alpha = 0)
        bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
        bsdf.location = (-300, 150)
        bsdf.inputs['Alpha'].default_value = 0.0

        # Emission Shader (strength 0, placeholder)
        emission = nodes.new(type='ShaderNodeEmission')
        emission.location = (-300, -50)
        emission.inputs['Color'].default_value = (1.0, 1.0, 1.0, 1.0)
        emission.inputs['Strength'].default_value = 0.0

        # Mix Shader (Fac = 0.5)
        mix = nodes.new(type='ShaderNodeMixShader')
        mix.location = (100, 50)
        mix.inputs['Fac'].default_value = 0.5

        # Connect BSDF and Emission into Mix
        links.new(bsdf.outputs['BSDF'], mix.inputs[1])
        links.new(emission.outputs['Emission'], mix.inputs[2])
        links.new(mix.outputs['Shader'], output.inputs['Surface'])

        mat.blend_method = 'BLEND'

    elif mat.tr7ae_blend_value == 6:
        nodes.clear()

        output = nodes.new(type='ShaderNodeOutputMaterial')
        output.location = (200, 0)

        # Keep the texture node around (but don't connect it)
        tex_image = nodes.new(type='ShaderNodeTexImage')
        tex_image.location = (-400, 0)
        if old_image:
            tex_image.image = old_image
        tex_image.image.colorspace_settings.name = "sRGB"

        # Invisible shader
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
        description="Blending mode used by the mesh",
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

        # Search all hspheres globally
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

        # Compute object position in bone-local space
        bone_matrix = arm.matrix_world @ arm.data.bones[bone_name].matrix_local
        local_pos = bone_matrix.inverted() @ obj.matrix_world.translation

        # Write into the target SphereInfo entry
        target.x = local_pos.x
        target.y = local_pos.y
        target.z = local_pos.z
        target.radius = obj.scale.x  # assumes uniform scaling on X for radius

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

        # Search all HBoxes globally
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

        # Compute local transform in bone space
        bone_mat = arm.matrix_world @ arm.data.bones[bone_name].matrix_local
        local_mat = bone_mat.inverted() @ obj.matrix_world

        # --- Update position (in bone-local space) ---
        loc = local_mat.to_translation()
        targetbox.positionboxx = loc.x
        targetbox.positionboxy = loc.y
        targetbox.positionboxz = loc.z

        # --- Update half-extents from scale ---
        targetbox.widthx = obj.scale.x
        targetbox.widthy = obj.scale.y
        targetbox.widthz = obj.scale.z
        # Optional: target.widthw = 0.0

        # --- Update rotation ---
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

        # extract capsule index from name: HCapsule_<bone>_<idx>
        try:
            cap_idx = int(obj.name.split("_")[-1])
        except:
            cap_idx = 0
        cap = pbone.tr7ae_hcapsules[cap_idx]

        # build rest‐space bone matrix
        bone_mat  = arm.matrix_world @ arm.data.bones[bone_name].matrix_local
        # compute local transform relative to bone
        local_mat = bone_mat.inverted() @ obj.matrix_world

        # --- Update position ---
        loc = local_mat.to_translation()
        cap.posx = loc.x
        cap.posy = loc.y
        cap.posz = loc.z

        # --- Update orientation quaternion ---
        quat = local_mat.to_quaternion()
        cap.quatx = quat.x
        cap.quaty = quat.y
        cap.quatz = quat.z
        cap.quatw = quat.w

        # --- Update radius & length from object’s scale ---
        # (we used scale=(radius, radius, length) on import)
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

        # Search all markers across all bones by global index
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

        # Get object's world matrix
        world_mat = obj.matrix_world

        # Convert to bone space
        bone_mat = arm.matrix_world @ arm.data.bones[bone_name].matrix_local
        local_mat = bone_mat.inverted() @ world_mat

        # Update position
        pos = local_mat.to_translation()
        marker.px = pos.x
        marker.py = pos.y
        marker.pz = pos.z

        # Update rotation (approximate Euler)
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

        # Bone rest matrix in world space
        bone_matrix_world = armature.matrix_world @ armature.data.bones[bone_name].matrix_local

        # Object world matrix (already parented to "Targets" empty)
        target_matrix_world = obj.matrix_world

        # Compute local transform relative to bone
        relative_matrix = bone_matrix_world.inverted() @ target_matrix_world

        loc = relative_matrix.to_translation()
        rot = relative_matrix.to_euler('XYZ')

        obj.tr7ae_modeltarget.px = loc.x
        obj.tr7ae_modeltarget.py = loc.y
        obj.tr7ae_modeltarget.pz = loc.z
        obj.tr7ae_modeltarget.rx = rot.x
        obj.tr7ae_modeltarget.ry = rot.y
        obj.tr7ae_modeltarget.rz = rot.z



def register():
    # Classes registration
    classes = [
        ClothPointData, DistRuleData, ClothJointMapData,
        TR7AE_SphereInfo, TR7AE_HCapsuleInfo, TR7AE_HBoxInfo, TR7AE_HMarkerInfo,
        TR7AE_OT_ImportModel, TR7AE_PT_Tools, TR7AE_OT_ToggleHSpheres,
        TR7AE_OT_ToggleHBoxes, TR7AE_OT_ToggleHMarkers, TR7AE_PT_DrawGroupPanel,
        TR7AE_PT_ModelDebugProperties, TR7AE_PT_MaterialPanel, TR7AE_PT_SphereInfo,
        TR7AE_PT_HCapsuleInfo, TR7AE_PT_HBoxInfo, TR7AE_PT_MarkersInfo, TR7AE_PT_ClothPointInspector, TR7AE_PT_DistRuleInspector, TR7AE_PT_ClothJointMapInspector,
        TR7AE_OT_NormalizeAndLimitWeights, TR7AE_OT_ToggleMarker, TR7AE_OT_ToggleHBox, TR7AE_OT_AddHBox, TR7AE_OT_AddHCapsule,
        TR7AE_OT_ToggleSphere, TR7AE_OT_ToggleCapsule, TR7AE_OT_ToggleBox, TR7AE_OT_AddHSphere, TR7AE_OT_AddHMarker,
        TR7AE_PT_HMarkerMeshInfo, TR7AE_PT_HCapsuleMeshInfo, TR7AE_OT_ToggleHCapsules, TR7AE_PT_AddHInfoButtons,
        TR7AE_OT_ExportCustomModel, TR7AE_OT_SnapBoneToHMarker, TR7AE_OT_ConvertImageToPCD, TR7AE_PT_TextureTools,
        TR7AE_PT_ClothPanel, TR7AE_ClothSettings, TR7AE_ModelTargetInfo, TR7AE_OT_ConvertPCDToDDS,
        TR7AE_PT_HSphereMeshInfo, TR7AE_PT_UtilitiesPanel, TR7AE_Preferences,
        TR7AE_PT_VisibilityPanel, TR7AE_SectionPaths, TR7AE_OT_ToggleHInfoVisibility,
        TR7AE_PT_FileSectionsPanel
    ]

    for cls in classes:
        bpy.utils.register_class(cls)

    # Properties registration
    bpy.types.PoseBone.tr7ae_hmarkers = bpy.props.CollectionProperty(type=TR7AE_HMarkerInfo)
    bpy.types.PoseBone.tr7ae_hcapsules = bpy.props.CollectionProperty(type=TR7AE_HCapsuleInfo)
    bpy.types.PoseBone.tr7ae_hspheres = bpy.props.CollectionProperty(type=TR7AE_SphereInfo)
    bpy.types.PoseBone.tr7ae_hboxes = bpy.props.CollectionProperty(type=TR7AE_HBoxInfo)
    bpy.types.Object.cloth_points = bpy.props.CollectionProperty(type=ClothPointData)
    bpy.types.Object.tr7ae_distrules = bpy.props.CollectionProperty(type=DistRuleData)
    bpy.types.Object.tr7ae_jointmaps = bpy.props.CollectionProperty(type=ClothJointMapData)

    bpy.types.Object.tr7ae_sections = bpy.props.PointerProperty(type=TR7AE_SectionPaths)
    bpy.types.Object.tr7ae_modeltarget = bpy.props.PointerProperty(type=TR7AE_ModelTargetInfo)
    bpy.types.Object.tr7ae_cloth = bpy.props.PointerProperty(type=TR7AE_ClothSettings)

    register_envmap_property()
    register_eyerefenvmap_property()
    register_draw_group_property()
    register_material_properties()
    register_panel_properties()

    # Handlers registration
    handlers = [
        tr7ae_sync_handler, sync_model_target_properties,
        sync_hmarker_transforms, sync_hbox_transforms,
        sync_hcapsule_transforms
    ]

    for handler in handlers:
        if handler not in bpy.app.handlers.depsgraph_update_post:
            bpy.app.handlers.depsgraph_update_post.append(handler)



def unregister():
    # Remove handlers
    handlers = [
        tr7ae_sync_handler, sync_model_target_properties,
        sync_hmarker_transforms, sync_hbox_transforms,
        sync_hcapsule_transforms
    ]

    for handler in handlers:
        if handler in bpy.app.handlers.depsgraph_update_post:
            bpy.app.handlers.depsgraph_update_post.remove(handler)

    # Unregister properties
    del bpy.types.PoseBone.tr7ae_hspheres
    del bpy.types.PoseBone.tr7ae_hcapsules
    del bpy.types.PoseBone.tr7ae_hboxes
    del bpy.types.PoseBone.tr7ae_hmarkers
    del bpy.types.Object.cloth_points
    del bpy.types.Object.tr7ae_distrules
    del bpy.types.Object.tr7ae_jointmaps

    del bpy.types.Object.tr7ae_sections
    del bpy.types.Object.tr7ae_modeltarget
    del bpy.types.Object.tr7ae_cloth

    unregister_envmap_property()
    unregister_draw_group_property()
    unregister_material_properties()
    unregister_panel_properties()

    # Classes unregistration (reverse order)
    classes = [
        TR7AE_PT_FileSectionsPanel, TR7AE_SectionPaths,
        TR7AE_PT_VisibilityPanel, TR7AE_PT_UtilitiesPanel,
        TR7AE_PT_HSphereMeshInfo, TR7AE_ModelTargetInfo,
        TR7AE_ClothSettings, TR7AE_PT_ClothPanel,
        TR7AE_OT_SnapBoneToHMarker, TR7AE_PT_HCapsuleMeshInfo,
        TR7AE_PT_HMarkerMeshInfo, TR7AE_OT_ToggleBox,
        TR7AE_OT_ToggleCapsule, TR7AE_OT_ToggleSphere, TR7AE_PT_ClothPointInspector, TR7AE_PT_DistRuleInspector, TR7AE_PT_ClothJointMapInspector,
        TR7AE_OT_ToggleMarker, TR7AE_PT_MarkersInfo, TR7AE_OT_AddHSphere,
        TR7AE_OT_ExportCustomModel, TR7AE_OT_NormalizeAndLimitWeights, TR7AE_OT_AddHCapsule,
        TR7AE_PT_HBoxInfo, TR7AE_PT_HCapsuleInfo, TR7AE_OT_ToggleHBox, TR7AE_OT_AddHMarker, TR7AE_OT_AddHBox,
        TR7AE_PT_SphereInfo, TR7AE_PT_MaterialPanel, TR7AE_OT_ConvertImageToPCD, TR7AE_PT_TextureTools,
        TR7AE_PT_ModelDebugProperties, TR7AE_PT_DrawGroupPanel, TR7AE_OT_ConvertPCDToDDS, TR7AE_PT_AddHInfoButtons,
        TR7AE_OT_ToggleHMarkers, TR7AE_OT_ToggleHBoxes, TR7AE_Preferences,
        TR7AE_OT_ToggleHSpheres, TR7AE_PT_Tools, TR7AE_OT_ToggleHCapsules,
        TR7AE_OT_ImportModel, TR7AE_HMarkerInfo, TR7AE_OT_ToggleHInfoVisibility,
        TR7AE_HBoxInfo, TR7AE_HCapsuleInfo, TR7AE_SphereInfo,
        ClothPointData, DistRuleData, ClothJointMapData
    ]

    for cls in classes:
        bpy.utils.unregister_class(cls)

if __name__ == "__main__":
    register()