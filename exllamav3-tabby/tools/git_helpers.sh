# Git helpers shared by setup and CPU regression checks. Requires die() and say().
preflight_repo() {
  local path="$1" status
  if [[ -e "$path" ]]; then
    [[ -e "$path/.git" ]] || die "$path exists but is not a repository root; choose a fresh source directory"
    status="$(git -C "$path" status --porcelain --untracked-files=all)" || die "$path is not a usable git checkout"
    [[ -z "$status" ]] \
      || die "$path has local changes or untracked files. Commit/stash them, or select a fresh source directory; setup will not overwrite them."
    [[ ! -L "$path/build" ]] || die "$path/build is a symlink; choose a fresh source directory"
    [[ -z "$(git -C "$path" ls-files build)" ]] || die "$path/build contains tracked files; refusing to treat it as a disposable build directory"
  fi
}
repo_key() {
  local url="${1%.git}"
  url="${url#https://}"; url="${url#http://}"; url="${url#ssh://}"
  url="${url#git@}"; url="${url/:/\/}"
  printf '%s' "${url,,}"
}
sync_repo() {
  local path="$1" wanted="$2" remote=origin old
  if [[ ! -e "$path" ]]; then
    say "cloning $wanted"
    git clone "$wanted" "$path" >&2 || die "clone failed: $wanted"
  fi
  old="$(git -C "$path" remote get-url origin 2>/dev/null || true)"
  if [[ "$(repo_key "$old")" != "$(repo_key "$wanted")" ]]; then
    remote=recipe
    if git -C "$path" remote get-url recipe >/dev/null 2>&1; then
      [[ "$(repo_key "$(git -C "$path" remote get-url recipe)")" == "$(repo_key "$wanted")" ]] \
        || die "$path remote 'recipe' already points elsewhere; choose a fresh source directory"
    else
      git -C "$path" remote add recipe "$wanted" || die "could not add recipe remote at $path"
      say "$path: preserving origin; fetching the configured fork through remote 'recipe'"
    fi
  fi
  git -C "$path" fetch -q "$remote" --tags || die "fetch failed at $path"
  printf '%s' "$remote"
}
resolve_ref() {
  local path="$1" remote="$2" ref="$3"
  git -C "$path" rev-parse --verify --end-of-options "refs/remotes/$remote/$ref^{commit}" 2>/dev/null \
    || git -C "$path" rev-parse --verify --end-of-options "$ref^{commit}"
}

