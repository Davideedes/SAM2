# ########## FOR JANNIKS FILES
# import os
# import shutil

# def move_images_from_list(folder_path, list_filename="schlagloecher.txt", target_folder_name="moved_images"):
#     txt_path = os.path.join(folder_path, list_filename)
#     target_folder = os.path.join(folder_path, target_folder_name)
#     os.makedirs(target_folder, exist_ok=True)

#     if not os.path.isfile(txt_path):
#         print(f"❌ File not found: {txt_path}")
#         return

#     with open(txt_path, "r", encoding="utf-8") as f:
#         image_names = [line.strip() for line in f if line.strip()]

#     for image_name in image_names:
#         source_path = os.path.join(folder_path, image_name)
#         destination_path = os.path.join(target_folder, image_name)

#         if os.path.isfile(source_path):
#             shutil.move(source_path, destination_path)
#             print(f"✅ Moved: {image_name}")
#         else:
#             print(f"⚠️ Not found: {image_name}")

# if __name__ == "__main__":
#     # Example: specify the path to the folder
#     folder = "seq/seq2_jannik"  # <--- change this
#     move_images_from_list(folder)
# #########################################################

##KES FILES