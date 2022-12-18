import os
import shutil
def rename_sequence(directory, out_directory, prefix):
    files = os.listdir(directory)

    if not os.path.exists(out_directory):
        os.makedirs(out_directory)

    # copy to new directory
    for f in files:
        new_name = f.split("_")[-1]
        shutil.copy(directory + "/" + f, out_directory + "/" + prefix + "_" + new_name)

if __name__ == '__main__':
    rename_sequence(
        "outputs/sequence-generation/pa1/crazy_monster_0.3_0_24.0_940_540/crazy_monster",
        "outputs/sequence-generation-processed/crazy_monster_0.3_0_24.0_940_540/crazy_monster"
        , "crazy_monster"
    )