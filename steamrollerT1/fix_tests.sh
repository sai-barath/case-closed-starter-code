#!/bin/bash
# Add TrailSet initialization after Trail definitions in test files
for file in *_test.go; do
    if [ -f "$file" ]; then
        # This is a simple fix - add TrailSet: make(map[Position]bool), after each Trail: line
        sed -i '/Trail:.*\[\]Position{{X:/a\TrailSet:        make(map[Position]bool),' "$file"
        sed -i '/Trail:.*make([]Position, /a\TrailSet:        make(map[Position]bool),' "$file"
    fi
done
