# Upstream Sync Guide

This document explains how to keep your LLM CLI fork synchronized with the upstream llama.cpp repository.

## Initial Setup

If you haven't already added the upstream remote:

```bash
git remote add upstream https://github.com/ggml-org/llama.cpp.git
git fetch upstream
```

Verify your remotes:
```bash
git remote -v
# Should show:
# origin    https://github.com/gnana997/llm.git (fetch)
# origin    https://github.com/gnana997/llm.git (push)
# upstream  https://github.com/ggml-org/llama.cpp.git (fetch)
# upstream  https://github.com/ggml-org/llama.cpp.git (push)
```

## Regular Sync Process

### 1. Fetch Latest Changes

```bash
git fetch upstream
git fetch origin
```

### 2. Sync Your Master Branch

```bash
# Switch to master branch
git checkout master

# Merge upstream changes
git merge upstream/master

# Push to your fork
git push origin master
```

### 3. Update Your Feature Branch

If you're working on a feature branch:

```bash
# Switch to your feature branch
git checkout your-feature-branch

# Rebase on updated master
git rebase master

# Force push if necessary (be careful!)
git push origin your-feature-branch --force-with-lease
```

## Handling Conflicts

### Common Conflict Areas

1. **CMakeLists.txt** - Build configuration changes
2. **ggml files** - Core library updates
3. **Documentation** - README, contributing guides

### Resolving Conflicts

1. When conflicts occur during merge:
   ```bash
   # See conflicted files
   git status
   
   # Edit each conflicted file
   # Look for conflict markers: <<<<<<< ======= >>>>>>>
   # Keep your CLI changes while incorporating upstream updates
   ```

2. After resolving conflicts:
   ```bash
   git add <resolved-files>
   git merge --continue
   ```

3. Test thoroughly:
   ```bash
   # Rebuild the project
   cmake -B build -DGGML_CUDA=ON
   cmake --build build --config Release -j
   
   # Test CLI functionality
   ./build/bin/llm --help
   ./build/bin/llm gpu-info
   ```

## Best Practices

### 1. Sync Frequently
- Sync at least weekly to minimize conflicts
- Always sync before starting new features

### 2. Keep Changes Isolated
- Keep CLI-specific changes in dedicated directories:
  - `tools/llm/`
  - `common/cli-framework/`
- Minimize changes to core llama.cpp files

### 3. Document Your Changes
- Maintain a CHANGELOG for CLI-specific features
- Comment your modifications clearly
- Update documentation when adding features

### 4. Test After Syncing
Always run tests after syncing:
```bash
# Run CLI tests
./test_gpu_info.sh

# Build and test different configurations
cmake -B build-cpu -DGGML_CUDA=OFF
cmake --build build-cpu --config Release

cmake -B build-cuda -DGGML_CUDA=ON
cmake --build build-cuda --config Release
```

## Automated Sync (Optional)

You can set up a GitHub Action to notify you of upstream changes:

1. Create `.github/workflows/upstream-check.yml`
2. Configure it to run daily and create issues for updates
3. This helps you stay aware of important changes

## Troubleshooting

### "refusing to merge unrelated histories"
```bash
git merge upstream/master --allow-unrelated-histories
```

### Large conflicts in generated files
- Consider resetting the file to upstream version
- Re-apply your changes manually

### Build failures after sync
1. Clean build directory: `rm -rf build/`
2. Check for new dependencies in upstream
3. Review upstream changelog for breaking changes

## Cherry-Picking Specific Changes

To apply only specific commits from upstream:

```bash
# Find the commit hash
git log upstream/master --oneline

# Cherry-pick the commit
git cherry-pick <commit-hash>
```

## Maintaining Fork Integrity

1. **Never force push to master**
2. **Keep a backup branch** before major syncs:
   ```bash
   git branch backup-before-sync
   ```
3. **Document upstream version** in your releases

## Questions?

If you encounter issues with syncing:
1. Check upstream llama.cpp issues
2. Open a discussion in our repository
3. Review the conflict resolution guide above

Remember: The goal is to benefit from upstream improvements while maintaining our CLI enhancements!