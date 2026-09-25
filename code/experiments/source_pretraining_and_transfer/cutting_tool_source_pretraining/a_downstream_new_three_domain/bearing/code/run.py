import sys
from pathlib import Path
PACKAGE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PACKAGE.parent/'shared_domain_components/bearing_code'))
from bearing_adapter_study import main,write
if __name__=='__main__':
    try:
        main(PACKAGE)
    except BaseException as error:
        write(PACKAGE/'status.json',dict(state='failed',error=repr(error)))
        raise
