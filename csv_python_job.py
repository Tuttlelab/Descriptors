import os
import subprocess
import time
import logging

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler('job_tracker.log'),
            logging.StreamHandler()
        ]
    )

def submit_job(input_dir):
    logging.info(f"Submitting job for directory: {input_dir}")
    cmd = ['sbatch', 'csv_script.sh', input_dir]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        job_id = result.stdout.strip().split()[-1]
        logging.info(f"Job submitted successfully - ID: {job_id}")
        return job_id
    else:
        logging.error(f"Job submission failed: {result.stderr}")
        return None

def check_job_status(job_id):
    # Check job status using squeue
    result = subprocess.run(['squeue', '-j', job_id],
                          capture_output=True,
                          text=True)
    return len(result.stdout.split('\n')) > 2  # If more than 2 lines, job is still running

def get_all_dirs():
    base_path = os.path.join(os.getcwd(), "centered_files")
    print(f"Scanning base path: {base_path}")
    dirs_to_process = []

    for category in ["high_ap", "mid_ap"]:
        category_path = os.path.join(base_path, category)
        print(f"Checking category path: {category_path}")

        if not os.path.exists(category_path):
            print(f"Category path does not exist: {category_path}")
            continue

        items = os.listdir(category_path)
        print(f"Found {len(items)} items in {category}")

        for folder in items:
            folder_path = os.path.join(category_path, folder)
            print(f"Checking folder: {folder_path}")

            if not os.path.isdir(folder_path):
                print(f"Not a directory: {folder_path}")
                continue

            # # Find any .gro and .xtc files
            # gro_files = [f for f in os.listdir(folder_path) if f.endswith('.gro')]
            # xtc_files = [f for f in os.listdir(folder_path) if f.endswith('.xtc')]

            # if gro_files and xtc_files:
            dirs_to_process.append(folder_path)
            #     print(f"✓ Added valid directory: {folder_path}")
            #     print(f"  Found GRO: {gro_files[0]}")
            #     print(f"  Found XTC: {xtc_files[0]}")
            # else:
            #     print(f"× Missing required files in: {folder_path}")
            #     print(f"  GRO files found: {len(gro_files)}")
            #     print(f"  XTC files found: {len(xtc_files)}")

    return dirs_to_process

def main():
    setup_logging()
    logging.info("Starting job submission process")

    # Get all directories to process
    dirs = get_all_dirs()
    logging.info(f"Found {len(dirs)} directories to process")

    active_jobs = {}
    failed_jobs = []

    # Submit jobs
    for dir_path in dirs:
        job_id = submit_job(dir_path)
        if job_id:
            active_jobs[job_id] = dir_path
            logging.info(f"Queued job {job_id} for {dir_path}")
            # time.sleep(10)
        else:
            failed_jobs.append(dir_path)
            logging.error(f"Failed to submit job for {dir_path}")

    # # Monitor jobs
    # while active_jobs:
    #     current_jobs = list(active_jobs.keys())
    #     for job_id in current_jobs:
    #         if not check_job_status(job_id):
    #             dir_path = active_jobs[job_id]
    #             logging.info(f"Job {job_id} for {dir_path} has completed!")
    #             del active_jobs[job_id]

    #     if active_jobs:
    #         logging.info(f"Waiting for {len(active_jobs)} jobs to complete...")
    #         time.sleep(60)

    # # Final report
    # logging.info("All jobs completed!")
    # if failed_jobs:
    #     logging.warning(f"Failed jobs for directories: {failed_jobs}")

if __name__ == "__main__":
    main()
