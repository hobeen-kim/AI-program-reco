import pandas as pd
from bs4 import BeautifulSoup


# HTML 태그를 제거하는 함수
def remove_html_tags(text):
    if isinstance(text, str) and ('<' in text and '>' in text):  # HTML 태그가 있는지 확인
        return BeautifulSoup(text, "html.parser").get_text()
    return text


# 파일 경로 설정
input_file_path = './files/program_raw.csv'
output_file_path = './files/refined_data.csv'

# CSV 파일을 읽어옴
df = pd.read_csv(input_file_path)

# 필요한 열만 선택
df_refined = df.loc[:, [
                           'title', 'activity_start_date', 'activity_end_date',
                           'detail_text', 'main_text',
                           'max_activity_day', 'max_per_team', 'min_activity_day',
                           'min_per_team', 'notice_end_at', 'notice_start_at', 'person_number',
                           'possible_child', 'team_number', 'participation_fee', 'supportive_child',
                           'pet', 'is_paid_program', 'fcfs', 'participation_fee_by_team', 'sido_name',
                           'sigg_name', 'is_full_subsidy', 'subsidy_for_1', 'subsidy_for_2', 'subsidy_for_3',
                           'subsidy_for_4', 'subsidy_for_5', 'transportation_subsidy', 'accommodation_subsidy', 'experience_subsidy',
                           'meal_subsidy', 'travel_subsidy', 'etc_subsidy', 'monthler_url'
                       ]]

# main_text 열에서 HTML 태그 제거
df_refined.loc[:, 'detail_text'] = df_refined['detail_text'].apply(remove_html_tags)

# True/False 값을 '가능'/'불가능'으로 변환
bool_columns = ['possible_child', 'supportive_child', 'pet', 'is_paid_program', 'fcfs',
                'participation_fee_by_team', 'is_full_subsidy', 'transportation_subsidy',
                'accommodation_subsidy', 'experience_subsidy', 'meal_subsidy', 'travel_subsidy']

# 각 열에 대해 True/False를 '가능'/'불가능'으로 변환
df_refined[bool_columns] = df_refined[bool_columns].applymap(lambda x: '가능' if x else '불가능')

# 열 이름 변경
df_refined = df_refined.rename(columns={
    # 'program_id': '프로그램 id',
    'monthler_url': '한달살러 링크',
    'title': '프로그램 제목',
    'activity_start_date': '활동 시작 시간',
    'activity_end_date': '활동 종료 시간',
    # 'announcement_at': '선정 발표 시간',
    # 'announcement_text': '선정 발표 공지',
    # 'published_at': '프로그램 게시일',
    # 'apply_url': '지원하는 url',
    'detail_text': '프로그램 상세 설명',
    'main_text': '프로그램 요약 설명',
    # 'manager_email': '담당자 이메일',
    # 'manager_phone': '담당자 전화번호',
    'max_activity_day': '최대 활동(여행)일수',
    'min_activity_day': '최소 활동(여행)일수',
    'max_per_team': '팀당 최대 인원',
    'min_per_team': '팀당 최소 인원',
    'notice_end_at': '공고(지원) 마감 시간',
    'notice_start_at': '공고(지원) 시작 시간',
    # 'notice_url': '공고 url',
    'person_number': '모집 인원 수',
    'possible_child': '아이 동반 가능(아이 함께 가기)',
    'team_number': '모집 팀 수',
    'participation_fee': '참가비',
    # 'is_monthler_form': '전용 신청서 여부(한달살러 신청서)',
    'supportive_child': '아이 지원금 지원 여부',
    'pet': '반려동물 동반 가능 여부',
    'is_paid_program': '유료 프로그램 여부',
    'fcfs': '선착순 여부',
    'participation_fee_by_team': '참가비가 팀 당 필요한지 여부',
    'sido_name': '광역시/도 이름',
    'sigg_name': '시/군/구 이름',
    'is_full_subsidy': '전액 지원 여부',
    'subsidy_for_1': '1인당 지원금',
    'subsidy_for_2': '2인당 지원금',
    'subsidy_for_3': '3인당 지원금',
    'subsidy_for_4': '4인당 지원금',
    'subsidy_for_5': '5인당 지원금',
    'transportation_subsidy': '교통비 지원 여부',
    'accommodation_subsidy': '숙박비 지원 여부',
    'experience_subsidy': '체험비 지원 여부',
    'meal_subsidy': '식비 지원 여부',
    'travel_subsidy': '여행비 지원 여부',
    'etc_subsidy': '기타 지원금 항목'
})

# 선택한 열만 있는 새로운 CSV 파일로 저장
df_refined.to_csv(output_file_path, index=False)

# 파일 저장 완료 메시지 출력
output_file_path
