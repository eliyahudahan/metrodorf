#!/bin/bash

# ============================================
# 📁 PART 1: FIND AND PRESERVE ALL EMPTY FOLDERS
# ============================================
echo "📁 Checking for empty folders..."

# Find all empty directories (excluding .git and __pycache__)
EMPTY_FOLDERS=$(find . -type d -empty -not -path "./.git/*" -not -path "*/__pycache__/*" -not -name "__pycache__")

if [[ -n "$EMPTY_FOLDERS" ]]; then
    echo "   Found empty folders. Adding .gitkeep to preserve structure:"
    
    # Loop through each empty folder and add .gitkeep
    echo "$EMPTY_FOLDERS" | while read folder; do
        echo "   📌 Preserving: $folder/"
        touch "$folder/.gitkeep"
    done
else
    echo "   ✅ No empty folders found."
fi

# ============================================
# 📦 PART 2: NORMAL GIT BACKUP
# ============================================
echo "📦 Checking git status..."
git status

# Save current changes to stash if any
if [[ -n $(git status --porcelain) ]]; then
    echo "💾 Stashing local changes..."
    git stash
    
    echo "🔄 Pulling latest changes with rebase..."
    git pull --rebase origin main
    
    echo "📤 Applying local changes back..."
    git stash pop
    
    # Check for conflicts
    if [[ -n $(git diff --name-only --diff-filter=U) ]]; then
        echo "⚠️  Conflicts detected! Please resolve manually."
        echo "Conflicting files:"
        git diff --name-only --diff-filter=U
        exit 1
    fi
    
    # Add all changes (including new files and .gitkeep files)
    echo "➕ Adding all changes..."
    git add .
    
    # Check if there's anything to commit after add
    if git diff --cached --quiet; then
        echo "✨ No changes to commit after adding."
        exit 0
    fi

    # Ask the user for a commit message
    echo "📝 Please enter your commit message:"
    read commit_message

    # Commit the changes with the given message
    echo "💾 Committing your changes..."
    git commit -m "$commit_message"

    # Push the changes to the 'main' branch
    echo "🚀 Pushing changes to 'main' branch..."
    git push -u origin main

    echo "✅ Backup completed successfully!"
else
    echo "✨ No changes detected. Pulling latest anyway..."
    git pull --rebase origin main
    echo "✅ Repository is up to date!"
fi