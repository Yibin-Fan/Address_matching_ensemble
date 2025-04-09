def extract_unique_addresses(input_file, output_file):
    unique_addresses = set()

    # Read and extract unique addresses
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            _, addr2, _ = line.strip().split('\t')
            unique_addresses.add(addr2)

    # Save unique addresses to file
    with open(output_file, 'w', encoding='utf-8') as f:
        for addr in sorted(unique_addresses):
            f.write(addr + '\n')

    return len(unique_addresses)


# Extract addresses
input_file = 'data/dataset/test/address.txt'
output_file = 'data/dataset/demo/unique_addresses.txt'
count = extract_unique_addresses(input_file, output_file)
print(f"Extracted {count} unique addresses to {output_file}")