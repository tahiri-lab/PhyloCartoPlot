# PhyloCartoPlot Documentation

Complete documentation for the PhyloCartoPlot workflow.

## Files

### 1. PIPELINE.md
**Technical documentation of the entire workflow**

Explains:
- Module breakdown (what each script does)
- Input/output specifications
- Data flow diagrams
- Key functions and their purposes
- Customization points
- Troubleshooting guide

**Read this for:** Understanding how the pipeline works technically

---

### 2. 01_phylocartoplot_walkthrough.ipynb
**Interactive step-by-step Jupyter notebook**

Walks through:
- Step 1: Format geographic coordinates
- Step 2: Add trait/metadata values
- Step 3: Build phylogenetic tree
- Step 4: Create visualization

**Read this for:** Hands-on learning, executing the workflow

#### Running the Notebook

```bash
# Navigate to docs folder
cd phylocartoplot_workflow/docs

# Start Jupyter
jupyter notebook

# Open: 01_phylocartoplot_walkthrough.ipynb
```

Or from project root:
```bash
jupyter notebook docs/01_phylocartoplot_walkthrough.ipynb
```

---

## How to Use This Documentation

### For Quick Understanding
1. Read the main **README.md** (in project root)
2. Check **STRUCTURE.txt** (in project root) for folder layout

### For Technical Details
1. Read **PIPELINE.md** (this folder)
2. Review script docstrings
3. Check comments in source code

### For Learning by Doing
1. Open **01_phylocartoplot_walkthrough.ipynb** (this folder)
2. Follow cells step-by-step
3. Execute and inspect outputs

### For Reference
- **PIPELINE.md** - Function details, customization
- **STRUCTURE.txt** - Folder organization, usage options

---

## Notebook Features

✅ Automatic path configuration
✅ Step-by-step explanations
✅ Data inspection and sampling
✅ Error checking and reporting
✅ Clear output messages
✅ Next step instructions

---

## Documentation Map

```
phylocartoplot_workflow/
├── README.md ........................ Project overview
├── STRUCTURE.txt .................... Folder layout & usage
├── SETUP_SUMMARY.md ................. Organization notes
├── CHANGES_SUMMARY.txt .............. Recent updates
│
└── docs/
    ├── README.md (this file) ........ Documentation index
    ├── PIPELINE.md .................. Technical details
    └── 01_phylocartoplot_walkthrough.ipynb ← START HERE (interactive)
```

---

## Quick Links

**New to PhyloCartoPlot?**
→ Start with README.md, then run the notebook in this folder

**Need technical details?**
→ Read PIPELINE.md or check source code

**Want to understand the structure?**
→ See STRUCTURE.txt in project root

**Ready to use the workflow?**
→ Run the notebook: `jupyter notebook docs/01_phylocartoplot_walkthrough.ipynb`

---

## The Tool is Generic!

PhyloCartoPlot now works with:
- ✅ Any phylogenetic tree
- ✅ Any species or taxa
- ✅ Any geographic region
- ✅ Any trait/metadata (just use `trait_value` column)
- ✅ Any raster or base map

See PIPELINE.md for customization details.
