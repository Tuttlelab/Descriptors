import os
import subprocess
import time

def submit_job(category, dipep_name, base_dir):
    # Submit the job using sbatch with parameters
    cmd = ['sbatch', 'script.sh', category, dipep_name, base_dir]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        job_id = result.stdout.strip().split()[-1]
        print(f"Job submitted for {category}/{dipep_name} with ID: {job_id}")
        return job_id
    else:
        print(f"Error submitting job: {result.stderr}")
        return None

def check_job_status(job_id):
    # Check job status using squeue
    result = subprocess.run(['squeue', '-j', job_id],
                          capture_output=True,
                          text=True)
    return len(result.stdout.split('\n')) > 2  # If more than 2 lines, job is still running

def get_dipep_dirs(base_dir, category):
    category_path = os.path.join(base_dir, category)
    return [d for d in os.listdir(category_path)
            if os.path.isdir(os.path.join(category_path, d))]

def main():
    base_dir = os.path.expanduser("~/Desktop/all_dipep_1200_60M")
    categories = ["high_ap", "mid_ap"]
    active_jobs = {}

    # Submit jobs for all directories
    for category in categories:
        dipep_dirs = get_dipep_dirs(base_dir, category)
        for dipep_name in dipep_dirs:
            job_id = submit_job(category, dipep_name, base_dir)
            if job_id:
                active_jobs[job_id] = f"{category}/{dipep_name}"

    # Monitor all jobs
    while active_jobs:
        for job_id in list(active_jobs.keys()):
            if not check_job_status(job_id):
                dipep_info = active_jobs[job_id]
                print(f"Job {job_id} for {dipep_info} has completed!")
                output_file = f"slurm-{job_id}.out"
                if os.path.exists(output_file):
                    print(f"Output available in: {output_file}")
                del active_jobs[job_id]

        if active_jobs:
            print(f"Waiting for {len(active_jobs)} jobs to complete...")
            time.sleep(60)

    print("All jobs completed!")

if __name__ == "__main__":
    main()
