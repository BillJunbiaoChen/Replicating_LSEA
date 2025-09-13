using Pkg
using Logging
global_logger(SimpleLogger(stderr, Logging.Error))

# Activate the environment and install packages according to the .toml files
println("Activating environment...")
Pkg.activate(".")
Pkg.resolve()

println("Installing packages...")
Pkg.instantiate()
Pkg.precompile()
println("Julia Environment setup is complete.")