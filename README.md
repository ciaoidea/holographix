import holo
from holo.field import Field
from holo.net.arch import content_id_bytes_from_uri

# Single-object encode / decode
holo.encode_image_holo_dir("frame.png", "frame.png.holo", target_chunk_kb=32)
holo.decode_image_holo_dir("frame.png.holo", "frame_recon.png")
holo.encode_audio_holo_dir("track.wav", "track.wav.holo", target_chunk_kb=32)
holo.decode_audio_holo_dir("track.wav.holo", "track_recon.wav")

# Photon-collector stacking
from holo.codec import stack_image_holo_dirs
stack_image_holo_dirs(["t0.png.holo", "t1.png.holo"], "stacked.png", max_chunks=8)

# Multi-object packing in one field
holo.pack_objects_holo_dir(["image1.jpg", "image2.jpg", "track.wav"], "scene.holo", target_chunk_kb=32)
holo.unpack_object_from_holo_dir("scene.holo", 0, "image1_rec.png")
holo.unpack_object_from_holo_dir("scene.holo", 2, "track_rec.wav")

# Field coverage + healing
f = Field("demo/image", "frame.png.holo")
print(f.coverage())
f.best_decode_image()                 # writes frame_recon.png
f.heal_to("frame_healed.holo")        # re-encodes current best view

# Build a content identifier (for transport/mesh)
cid = content_id_bytes_from_uri("holo://demo/image")
