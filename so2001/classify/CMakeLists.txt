# classify: tiny standalone tool

add_executable(classify classify.c)
zslm_apply_common_flags(classify)
# classify\sources sets no /arch flag at all; let MSVC defaults apply.
