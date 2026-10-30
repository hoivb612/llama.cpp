# Pull the Xbox_uGC_StaticBins vPack (prebuilt onnxruntime.lib + Rust tokenizers_c.lib,
# flat ORT headers, and runtime DLLs) into this directory's own object output, where every
# OrtPlugin consumer references it. Destination is $(OBJ_PATH)\$(O)\drop so all writes stay
# inside OBJ_PATH\$O as required by the build analyzer (ABT3000), out of $(_NTTREE) (ABT2904).
# This resolves to $(OBJECT_ROOT)\gamecore\so2001\OrtPlugin\Phi3\OrtDrop\objfre\amd64\drop.
# The payload expands as <dest>\OrtDrop\{bin,inc,lib}. /StrictVersioning with the pinned
# version [0.0.2] is required by the analyzer (ABT4600). vpack pull no-ops when the pinned
# version is already present; a clean build wipes the object tree, so it is re-pulled in pass 0.

PullOrtDrop:
	vpack pull /Name:Xbox_uGC_StaticBins /DestinationDirectory:$(OBJ_PATH)\$(O)\drop /VersionRange:[0.0.2] /StrictVersioning
