import os
import subprocess
from pathlib import Path

def convert_adni_to_flat_nii(input_adni_dir, output_nii_dir):
    """
    Walks through nested ADNI folders, converts DICOMs using dcm2niix,
    and saves them directly into a flat output folder inside the project.
    """
    # Define absolute paths
    base_project_dir = Path(__file__).parent.resolve()
    input_path = Path(input_adni_dir).resolve()
    output_path = base_project_dir / output_nii_dir
    
    # Create the output directory inside your project folder if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Scanning ADNI folder: {input_path}")
    print(f"Outputting flat NIfTI files to: {output_path}\n")
    
    # Track directories that actually contain .dcm files
    dicom_folders = set()
    for root, dirs, files in os.walk(input_path):
        if any(f.lower().endswith('.dcm') for f in files):
            dicom_folders.add(Path(root))
            
    if not dicom_folders:
        print("No DICOM (.dcm) files found in the specified input directory!")
        return

    print(f"Found {len(dicom_folders)} image series folders to convert.")
    
    # dcm2niix naming convention flags:
    # %n = Patient Name (In ADNI, this is usually the Subject ID like 002_S_0295)
    # %d = Description (The scan protocol sequence name like MPRAGE, t1_weighted)
    # %t = Time/Date of scan
    # %i = Image ID (Unique identifier number from ADNI)
    # Resulting name example: 002_S_0295_MPRAGE_20261108143022_I123456.nii.gz
    naming_format = "%n_%d_%t"
    
    for idx, dcm_dir in enumerate(dicom_folders, 1):
        print(f"[{idx}/{len(dicom_folders)}] Converting: {dcm_dir.relative_to(input_path)}")
        
        # Build the dcm2niix execution command
        # -z y : enables maximum GZ compression (.nii.gz)
        # -f   : specifies the file naming pattern
        # -o   : sets the output folder target
        cmd = [
            "dcm2niix",
            "-z", "y",
            "-f", naming_format,
            "-o", str(output_path),
            str(dcm_dir)
        ]
        
        try:
            # Run conversion silently unless an error occurs
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Error converting folder {dcm_dir}: {e.stderr.decode().strip()}")
        except FileNotFoundError:
            print("❌ Error: 'dcm2niix' is not installed or not found in your system PATH.")
            return

    print(f"\n Success! All converted files are located in: {output_path}")

if __name__ == "__main__":
    # Change this to the path where your raw downloaded ADNI directory lives
    # It can be a relative path or an absolute path (e.g., "C:/Users/.../Downloads/ADNI")
    in_root = r"C:\Users\Owner\Downloads\test_set_cn1_dataset\ADNI"
    
    # The name of the flat folder that will be automatically built inside your project folder
    out_root = "output_niftis_cn_test"
    
    convert_adni_to_flat_nii(in_root, out_root)
