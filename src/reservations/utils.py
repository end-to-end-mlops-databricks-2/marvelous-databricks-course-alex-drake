def get_package_version():
    with open("version.txt", "r") as f:
        version = f.readline()
    return version