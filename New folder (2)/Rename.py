import os
 
# Function to rename multiple files
def main():
   
    folder = "Vertical_Bar_Chart"
    for count, filename in enumerate(os.listdir(folder)):
        dst = f"VerticalBar{str(count)}.jpg"
        src =f"{folder}/{filename}"  # foldername/filename, if .py file is outside folder
        dst =f"{folder}/{dst}"
         
        # rename() function will
        # rename all the files
        os.rename(src, dst)
 
# Driver Code
if __name__ == '__main__':
     
    # Calling main() function
    main()
