"""Gate verification for Phase 2: Schema knowledge base retrieval."""

from src.retrieval import retrieve_schema_context

def main():
    test_cases = [
        ("Which customers placed the most orders?", {"customers", "orders"}),
        ("What are the top 5 products by unit price?", {"products"}),
        ("Show total sales revenue for each category in 2024", {"categories", "products", "orders", "order_items"}),
        ("List all suppliers located in Germany", {"suppliers"}),
        ("Which employees report to each manager?", {"employees"}),
        ("How much payment has been collected per payment method?", {"payments"}),
        ("Find customers who purchased laptop products", {"customers", "orders", "order_items", "products"}),
        ("List orders that are cancelled or not yet shipped", {"orders"}),
    ]

    all_passed = True
    for q, expected_tables in test_cases:
        ctx = retrieve_schema_context(q)
        retrieved_set = set(ctx.tables)
        missing = expected_tables - retrieved_set
        if missing:
            print(f"FAIL: '{q}'")
            print(f"   Expected: {expected_tables}")
            print(f"   Retrieved: {retrieved_set}")
            print(f"   Missing: {missing}")
            all_passed = False
        else:
            print(f"PASS: '{q}' -> Retrieved tables: {sorted(list(retrieved_set))}")

    assert all_passed, "Phase 2 Gate FAILED: Not all required tables were retrieved!"
    print("\nPhase 2 Gate Passed! All 8 manual test cases successfully retrieved all required tables.")

if __name__ == "__main__":
    main()

