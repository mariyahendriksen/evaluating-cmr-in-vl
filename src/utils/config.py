import os, fnmatch

def findReplace(directory, find, replace, filePattern):
    for path, dirs, files in os.walk(os.path.abspath(directory)):
        for filename in fnmatch.filter(files, filePattern):
            filepath = os.path.join(path, filename)
            # print(filepath)
            with open(filepath) as f:
                s = f.read()
            # print(s)
            s = s.replace(find, replace)
        # print('NEW:\n', s)
            with open(filepath, "w") as f:
                f.write(s)
