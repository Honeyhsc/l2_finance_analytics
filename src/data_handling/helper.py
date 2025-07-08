def reload_custom_modules(module_names=None, verbose=True, aggressive=False):
    """
    Reloads Python modules deeply with verbose output and aggressive submodule clearing.

    Args:
        module_names: List of short module names (e.g., ['trade_features'])
        verbose: Print detailed reloading info
        aggressive: Remove submodules before reimporting
    """
    import sys, os
    import importlib
    import inspect
    from pathlib import Path

    def get_project_root(target_dirname='project-root'):
        """
        Trim current path back to the project root folder (named `target_dirname`).
        Works on Windows and Unix paths.
        """
        current = Path.cwd().resolve()
        for parent in [current] + list(current.parents):
            if parent.name == target_dirname:
                return parent
        raise RuntimeError(f"❌ Could not find project root folder named '{target_dirname}' in path {current}")


    # --- CONFIG ---
    root = get_project_root() 
    src_paths = [
        root,
        root / 'src',
        root / 'src' / 'visualization',
        root / 'src' / 'data_handling',
        root / 'src' / 'features'
    ]

    # --- Set up sys.path ---
    for p in src_paths:
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))

    # --- Discover all available modules by stem name ---
    discovered_modules = {}
    for src in src_paths:
        print(f"🔍 Scanning for modules in: {src}")
        for py_file in src.rglob("*.py"):
            if py_file.stem.startswith("_"): continue
            print(f"🔍 Discovered module: {py_file}")
            rel_path = py_file.relative_to(root).with_suffix("")
            dotted_path = ".".join(rel_path.parts)
            discovered_modules[py_file.stem] = dotted_path

    # --- Use all discovered modules if none explicitly listed ---
    if module_names is None:
        default_short_names = [
            'plot_trade_features',
            'plot_distribution',
            'trade_lob_data_processing',
            'trade_features'
        ]
        module_names = [
            name for name in default_short_names if name in discovered_modules
        ]
        missing = [name for name in default_short_names if name not in discovered_modules]

        if verbose:
            print(f"🔍 No specific modules provided. Using default: {module_names}")
            if missing:
                print(f"⚠️ These modules were not found and will be skipped: {missing}")

    reloaded, failed = [], []

    # --- Optional: configure Jupyter autoreload ---
    if 'ipykernel' in sys.modules:
        try:
            get_ipython().run_line_magic('load_ext', 'autoreload')
            get_ipython().run_line_magic('autoreload', '3' if aggressive else '2')
            if verbose:
                print(f"\n⚙️ Jupyter autoreload enabled (aggressive={aggressive})")
        except:
            pass

    for short_name in module_names:
        import_path = discovered_modules.get(short_name)
        if not import_path:
            failed.append((short_name, "Not found in src paths"))
            if verbose:
                print(f"❌ Module not found: {short_name}")
            continue

        try:
            if verbose:
                print(f"\n🔁 Reloading {short_name} ({import_path})")

            # Aggressively remove submodules
            if aggressive:
                to_remove = [name for name in sys.modules if name.startswith(import_path)]
                for name in to_remove:
                    del sys.modules[name]
                if verbose and to_remove:
                    print(f"  - Removed from sys.modules: {to_remove}")

            # Fresh import
            mod = importlib.import_module(import_path)

            # Update top-level globals
            try:
                ipython = get_ipython()
                ipython.user_ns.update({k: v for k, v in mod.__dict__.items() if not k.startswith('_')})
            except:
                globals().update({k: v for k, v in mod.__dict__.items() if not k.startswith('_')})


            reloaded.append(short_name)
            if verbose:
                print(f"✅ Reloaded: {short_name}")
                print(f"  - File: {getattr(mod, '__file__', 'unknown')}")

        except Exception as e:
            failed.append((short_name, str(e)))
            if verbose:
                print(f"❌ Failed to reload {short_name}: {e}")

    # --- Final summary ---
    if verbose:
        print("\n📊 Reload Summary:")
        print(f"  ✔️ Reloaded ({len(reloaded)}): {', '.join(reloaded)}")
        if failed:
            print(f"  ❌ Failed ({len(failed)}):")
            for name, err in failed:
                print(f"    - {name}: {err}")
        print(f"  📦 Total processed: {len(module_names)}")

    return reloaded, failed
