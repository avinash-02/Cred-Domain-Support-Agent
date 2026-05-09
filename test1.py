def attendance_report(names, attendance):
    for name in names:
        count = attendance.get(name, 0)
        print(f"{name}: {count}")

    # Create and return list of names with attendance >= 3
    return [name for name in names if attendance.get(name, 0) >= 3]


# Sample function call with sample data
if __name__ == "__main__":
    names_list = ["Ram", "Shyam", "Raj", "Rani", "Kumar", "Rajiv"]

    attendance_dict = {
        "Ram": 5,
        "Shyam": 2,
        "Raj": 3,
        "Rani": 1,
        "Kumar": 4,
        # Rajiv" is intentionally missing to test .get() default
    }

    print("Attendance Report:")
    print("-" * 20)

    result = attendance_report(names_list, attendance_dict)
    
    print("-" * 20)
    print(f"Names with attendance >= 3: {result}")


