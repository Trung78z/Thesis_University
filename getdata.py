from roboflow import Roboflow
rf = Roboflow(api_key="cowrvvMbAWTLZjZnz40A")
project = rf.workspace("test-y9all").project("p2_dhaka_dataset-f6ba6-bwwcl")
version = project.version(1)
dataset = version.download("yolov11")
                