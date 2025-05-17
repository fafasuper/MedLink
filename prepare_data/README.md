> 😁描述：该目录用于处理XXX数据集，输入数据是[XXX](http://)，输出数据是XXX，XXX和XXX。可通过[XXX](http)下载该数据。

### 数据格式介绍和示例
#### 输入数据
> 
#### 输出数据
> 1. 输出的数据集包括以下5种：  
>  * MIMIC-3-full
>  * MIMIC-3-top50
>  * MIMIC-3-clean
>  * MIMIC-4-icd-9
>  * MIMIC-4-icd-10



<!--### mimic-iii和mimic-iv数据预处理过程(mimic-iii和mimic-iv原始数据需要自行在Physionet.org 数据仓库中注册下载)-->

#### 1. 预处理数据（参考caml的预处理步骤——Mullenbach的工作）
##### 1.1 输入文件
> 1. 下载数据：[Physionet.org](https://physionet.org/content/mimiciii/) 数据仓库中下载。（需要注册并接受培训才可以下载该数据）
> 2. 得到疾病诊断数据：DIAGNOSES_ICD.csv.gz
> 3. 得到手术程序的ICD-9代码数据：PROCEDURES_ICD.csv.gz
> 4. 得到出院小结数据：NOTEEVENTS.csv.gz
> 5. 患者出院数据：discharge.csv.gz：  

#### 1.2 获取MIMIC-3-full，MIMIC-3-top50
> * 首先，通过`prepare_mimiciii_mullenbach.py`文件获得`mimiciii_full.feather`，然后通过`generate_mimiciii.py`进行train，dev，test的划分获得`mimiciii_full_splits.feather`
> * 处理步骤：转换为小写、移除特殊字符、数字等.
> * 注意事项：top50的数据如果按照最新的mimic-iii版本来筛选，会和caml项目中获取的有些出入（Edin作者也发现这个问题，并分析可能是数据版本的问题），因此参考Edin的工作，也直接使用caml项目中提取的TOP50编码，不再自行获取。

#### 1.3 获取MIMIC-3-clean
> * 与步骤1.2雷同  
> * 数据处理步骤比mimic3full和50多了一步,即删除频率小于10的code  
> * 使用 download_and_preprocess_code_systems 函数下载和预处理疾病诊断和手术程序的 ICD-9 代码  
> * 移除重复代码和格式化代码  
> * 重新格式化诊断和手术的 ICD-9 代码
> * 格式和合并出院小结
> * 对数进行重命名，排序和索引重置
> * 使用 filter_codes 函数确保只包括出现次数超过最小阈值的代码
> * 将出小结和 ICD-9 代码数据框架合并
> * 使用本预处理器对文档进行预处理，包括转换为小写、移除特殊字符和数字等  

#### 1.4 获取MIMIC-4-icd-9, MIMIC-4-icd-10
> * 与步骤1.2雷同
> * 这里包括对 ICD 代码的版本处理，区分了 ICD-9 和 ICD-10 版本的代码  
> * 输入文件还包括discharge.csv.gz，包含患者的出院记录，主要用于获取患者的文本数据（出院小结）  
> * 对 ICD 代码进行格式化，添加小数点和处理版本（ICD-9 或 ICD-10），以及对应的诊断或手术标志
> * 通过parse_codes_dataframe 函数处理和重新格式化 ICD 代码数据，包括去重和分组  
> * 使用 parse_notes_dataframe 函数格式化出院记录数据框架，包括去重和缺失值处理
> * 合并数据：根据 ICD 版本将手术和诊断数据分为两组，一组为 ICD-9，另一组为 ICD-10。将处理后的 ICD 代码数据与出院记录数据按照患者 ID 进行合并
> * 筛选和预处理：筛选出至少出现10次的 ICD 代码。将缺失的 ICD 代码字段替换为空列表。定义目标列，结合所有的诊断和手术代码。删除没有任何 ICD 代码的记录。对出院小结进行文本预处理


##### 1.5 输出文件 （xxx.feather）
> 1. 出院摘要文本（text列）: 患者的出院摘要信息,简称出院小结
> 2. 患者ID （subject_id）和住院ID（_id）: 用于识别不同患者和其住院记录的唯一标识符
> 3. 医疗编码（target列）: 每个出院摘要对应的医疗编码，包括诊断编码（xx_proc）和手术编码（xx_diag），也是需要预测的类别



