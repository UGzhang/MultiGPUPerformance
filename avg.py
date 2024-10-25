def read_data_3(file_path):
    gpus = []
    times = []
    perfs = []
    index = 0
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
                    index=index+1
                if index >= 5 : break
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except ValueError:
        print("Error: Invalid data format. Ensure the file contains integers and floats.")
    
    return gpus, times, perfs

def read_data_2(file_path):
    gpus = []
    perfs = []
    index = 0
    try:
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.split()
                if len(parts) >= 1:
                    gpu = int(parts[0])
                    perf = float(parts[1])
                    gpus.append(gpu)
                    perfs.append(perf)
                    index=index+1
                if index >= 5 : break
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except ValueError:
        print("Error: Invalid data format. Ensure the file contains integers and floats.")
    
    return gpus, perfs

def calculate_average(times):
    if not times:
        return 0
    return sum(times) / len(times)

# def calculate_minimum(values):
#     if not values:
#         return None
#     return min(values)

def main():
    base_path = 'mpi-int/'  # Base path for the files
    file_extension = '.txt'
    
    for i in range(4,17):
        file_path = f"{base_path}{20480}{'-100-'}{i}{file_extension}"
        # _, times, perfs = read_data(file_path)
        _, perfs = read_data_2(file_path)
        
        if perfs:
            # average_time = calculate_minimum(times)
            average_perf = calculate_average(perfs)
            # print(f"File {file_path}: {average_time:.4f} {average_perf:.4f}")
            print(f"File {file_path}: {average_perf:.4f}")
        else:
            print(f"File {file_path}: No data available to calculate the average time.")

if __name__ == "__main__":
    main()
