import json
from collections import defaultdict

def print_entity_linking_sample(file_path, line_number=1):

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for i, line in enumerate(file, 1):
                if i == line_number:
                    data = json.loads(line)
                    print("line:", i)
                    print("\n:")
                    for ann in data.get('annotations', []):
                        print("-" * 50)
                        print(f"Mention: {ann.get('mention', 'N/A')}")
                        print(f"Entity Name: {ann.get('entity_name', 'N/A')}")
                        print(f"CUI: {ann.get('cui', 'N/A')}")
                        print(f"Types: {ann.get('types', [])}")
                        print(f"Start: {ann.get('start', 'N/A')}")
                        print(f"End: {ann.get('end', 'N/A')}")
                    break
            else:
                print(f"not found {line_number} 行")
    except FileNotFoundError:
        print("not found")
    except json.JSONDecodeError:
        print("JSON wrong")

def count_entity_types(file_path, output_path):

    type_counter = defaultdict(int)
    processed_types = set()
    
    print("processing...")
    with open(file_path, 'r', encoding='utf-8') as file:
        count = 1
        for line in file:
            if count % 1000 == 0:
                print(f"processed {count} lines")
            count += 1
            
            try:
                data = json.loads(line)
                annotations = data.get('annotations', [])
                
                for annotation in annotations:
                    # 获取types列表
                    types = annotation.get('types', [])
                    if not isinstance(types, list):
                        continue
                    
                    # 统计每个类型
                    for entity_type in types:
                        if entity_type:
                            type_counter[entity_type] += 1
                            processed_types.add(entity_type)
                            
            except json.JSONDecodeError as e:
                print(f" {count} line json wrong: {e}")
                continue
            except Exception as e:
                print(f" {count} line process wrong: {e}")
                continue


    sorted_types = sorted(processed_types, key=lambda x: type_counter[x], reverse=True)
    

    type_to_id = {t: i for i, t in enumerate(sorted_types)}
    

    with open(output_path, 'w', encoding='utf-8') as outfile:
        json.dump(type_to_id, outfile, ensure_ascii=False, indent=2)
    


    for t in list(sorted_types)[:10]:
        print(f"{t}: {type_counter[t]}")
    

    stats_output_path = output_path.replace('.json', '_with_stats.json')
    type_stats = {
        t: {
            'id': type_to_id[t],
            'count': type_counter[t]
        } for t in sorted_types
    }
    with open(stats_output_path, 'w', encoding='utf-8') as outfile:
        json.dump(type_stats, outfile, ensure_ascii=False, indent=2)
    
    return type_to_id

if __name__ == "__main__":
    
    file_path = "./random_sample_20K.jsonl"
    output_path = "./random_entity_type_to_id_20ksamples.json"
    

     print_entity_linking_sample(file_path, 1)
    

    type_to_id = count_entity_types(file_path, output_path)


