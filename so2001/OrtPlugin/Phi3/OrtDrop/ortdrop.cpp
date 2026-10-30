// Placeholder translation unit for the OrtDrop staging library.
//
// This directory's real payload is the Xbox_uGC_StaticBins vPack, pulled in pass 0 by the
// PullOrtDrop recipe in makefile.inc. The LIBRARY target type is only here so that $(O)
// resolves to the architecture subdirectory inside makefile.inc (a NOTARGET dir would leave
// $(O) empty). The resulting ortdrop.lib is intentionally unused; this function exists solely
// to give the translation unit an exported symbol and avoid an empty-object linker warning
// (LNK4221).
extern "C" int OrtDropPlaceholder()
{
    return 0;
}
