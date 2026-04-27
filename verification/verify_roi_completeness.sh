for d in data/lipread_roi/*/; do
    word=$(basename "$d")
    train=$(find "$d/train" -name "*.npz" 2>/dev/null | wc -l)
    val=$(find "$d/val" -name "*.npz" 2>/dev/null | wc -l)
    test=$(find "$d/test" -name "*.npz" 2>/dev/null | wc -l)
    if [ $train -eq 0 ] || [ $val -eq 0 ] || [ $test -eq 0 ]; then
        echo "INCOMPLETE: $word train=$train val=$val test=$test"
    fi
done
