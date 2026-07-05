$files = Get-ChildItem -Recurse -File
foreach ($file in $files) {
    try {
        $content = Get-Content $file.FullName -Raw
        if ($content -and $content.Contains("ARIA")) {
            $newContent = $content.Replace("ARIA", "ARIA")
            Set-Content $file.FullName $newContent -Encoding UTF8
        }
    } catch {
        Write-Warning "Could not process file $($file.FullName): $($_.Exception.Message)"
    }
}

