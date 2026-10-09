"""physedit: one command to generate and evaluate a set of video models on a PhysEdit benchmark.

config.py   the fleet (sites, hosts, GPUs), the models (how each is generated) and the benchmarks
driver.py   the pipeline: manifests -> generation -> copy home -> publish -> evaluation stages -> table
fleet.py    commands, detached jobs and copies on the hosts, issued from the hub
build.py    builds a model's manifests on the site that runs it
survey.py   reads a site's state for the driver
runners/    the generation scripts, deployed to every site
"""
