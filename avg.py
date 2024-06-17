def read_data(file_path):
    gpus = []
    times = []
    perfs = []
    
    try:
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.split()
                if len(parts) >= 2:
                    gpu = int(parts[0])
                    time = float(parts[1])
                    perf = float(parts[2])
                    gpus.append(gpu)
                    times.append(time)
                    perfs.append(perf)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except ValueError:
        print("Error: Invalid data format. Ensure the file contains integers and floats.")
    
    return gpus, times, perfs

def calculate_average(times):
    if not times:
        return 0
    return sum(times) / len(times)

def main():
    base_path = 'nvshmem/data/10240-100-'  # Base path for the files
    file_extension = '.txt'
    
    for i in range(1, 9):
        file_path = f"{base_path}{i}{file_extension}"
        _, times, perfs = read_data(file_path)
        
        if times:
            average_time = calculate_average(times)
            average_perf = calculate_average(perfs)
            print(f"File {file_path}: The average time is {average_time:.4f} {average_perf:.4f}")
        else:
            print(f"File {file_path}: No data available to calculate the average time.")

if __name__ == "__main__":
    main()
